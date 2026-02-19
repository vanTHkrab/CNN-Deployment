"""Service layer – Grad-CAM / CAM heatmap generation.

*  **Keras models** → true Grad-CAM (gradient-weighted class activation
   mapping) using ``tf.GradientTape``.
*  **TFLite models** → CAM (class activation mapping) by reading the
   last conv feature map and the Dense layer weights directly from the
   interpreter – **no ``.keras`` fallback required**.

   For ``GlobalAveragePooling → Dense`` architectures the CAM result is
   mathematically equivalent to Grad-CAM.

Both paths produce a heatmap overlay saved as a PNG in ``GRADCAM_DIR``.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.app.config import CLASS_NAMES, GRADCAM_DIR

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════

def _overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay a [0, 1]-normalised *heatmap* on *original_image* (jet palette)."""
    import matplotlib.cm as cm

    w, h = original_image.size
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), Image.BILINEAR),
    ).astype(np.float32) / 255.0

    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # drop alpha
    heatmap_colored = np.uint8(heatmap_colored * 255)

    heatmap_img = Image.fromarray(heatmap_colored)
    return Image.blend(original_image.convert("RGB"), heatmap_img, alpha=alpha)


# ══════════════════════════════════════════════
#  Keras  –  true Grad-CAM
# ══════════════════════════════════════════════

def _find_last_conv_layer_keras(model: tf.keras.Model) -> str | None:
    """Recursively find the last Conv2D / DepthwiseConv2D / SeparableConv2D."""
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )

    def _collect(layer: tf.keras.layers.Layer) -> list[tf.keras.layers.Layer]:
        layers: list[tf.keras.layers.Layer] = []
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                layers.extend(_collect(sub))
        else:
            layers.append(layer)
        return layers

    for layer in reversed(_collect(model)):
        if isinstance(layer, conv_types):
            return layer.name
    return None


def _find_layer_in_model(
    m: tf.keras.Model, name: str,
) -> tf.keras.layers.Layer | None:
    """Search for a layer by name, recursing into sub-models."""
    for layer in m.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, "layers"):
            found = _find_layer_in_model(layer, name)
            if found is not None:
                return found
    return None


def _compute_heatmap_keras(
    model: tf.keras.Model,
    img_array: np.ndarray,
    predicted_idx: int,
    conv_layer_name: str,
) -> np.ndarray:
    """
    Grad-CAM via GradientTape – **Keras 3 compatible**.

    Strategy for finetuned (nested backbone) architectures:
      1. Build an extractor from the **backbone's own graph**
         (both ``conv_layer.output`` and ``backbone.output`` belong to the
         same graph, so Keras 3 accepts it).
      2. Replay head layers eagerly inside ``GradientTape``.
      3. Compute gradients of the target score w.r.t. conv output.

    Returns a 2-D numpy array (H, W) with values in [0, 1].
    """
    target_layer = _find_layer_in_model(model, conv_layer_name)
    if target_layer is None:
        raise ValueError(f"Layer '{conv_layer_name}' not found in model.")

    img_tensor = tf.cast(img_array, tf.float32)

    # ── Find the backbone sub-model that owns the conv layer ──
    sub_model: tf.keras.Model | None = None
    head_start_idx: int = 0

    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "layers"):
            if _find_layer_in_model(layer, target_layer.name) is not None:
                sub_model = layer
                head_start_idx = idx + 1
                break

    if sub_model is not None:
        # Keras 3: sub_model.output may be a list → unwrap
        backbone_output = sub_model.output
        if isinstance(backbone_output, list):
            backbone_output = backbone_output[0]

        backbone_extractor = tf.keras.Model(
            inputs=sub_model.input,
            outputs=[target_layer.output, backbone_output],
        )

        with tf.GradientTape() as tape:
            conv_output, backbone_out = backbone_extractor(
                img_tensor, training=False,
            )
            tape.watch(conv_output)
            x = backbone_out
            for lyr in model.layers[head_start_idx:]:
                x = lyr(x)
            target_score = x[:, predicted_idx]

        grads = tape.gradient(target_score, conv_output)
    else:
        # Flat model (conv layer is a top-level child)
        flat_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_output, predictions = flat_extractor(
                img_tensor, training=False,
            )
            tape.watch(conv_output)
            target_score = predictions[:, predicted_idx]
        grads = tape.gradient(target_score, conv_output)

    if grads is None:
        raise RuntimeError("Gradients could not be computed.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (conv_output[0] @ pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


# ══════════════════════════════════════════════
#  TFLite  –  CAM (no gradients required)
# ══════════════════════════════════════════════

def _find_last_conv_tensor_tflite(
    interpreter: tf.lite.Interpreter,
) -> dict | None:
    """
    Find the last spatial feature-map tensor ``(1, H, W, C)`` in the
    TFLite graph (highest tensor index with ``H > 1`` and ``W > 1``).
    """
    best: dict | None = None
    for t in interpreter.get_tensor_details():
        shape = t["shape"]
        if len(shape) == 4 and shape[1] > 1 and shape[2] > 1:
            if best is None or t["index"] > best["index"]:
                best = t
    return best


def _find_dense_weights_tflite(
    interpreter: tf.lite.Interpreter,
    num_classes: int,
) -> np.ndarray | None:
    """
    Find the Dense weight matrix and return it as ``(features, num_classes)``.

    TFLite may store the kernel in either orientation:
    * ``(num_classes, features)``  – typical for TFLite converted models
    * ``(features, num_classes)``  – less common

    We look for a 2-D tensor where **one** dimension equals *num_classes*
    and the other is larger.  The result is always transposed (if needed)
    so the caller gets ``(features, num_classes)``.
    """
    best: dict | None = None
    best_transposed: bool = False

    for t in interpreter.get_tensor_details():
        shape = t["shape"]
        if len(shape) != 2:
            continue

        transposed = False
        if shape[0] == num_classes and shape[1] > num_classes:
            # (num_classes, features) — needs transpose
            transposed = True
        elif shape[1] == num_classes and shape[0] > num_classes:
            # (features, num_classes) — already correct
            transposed = False
        else:
            continue

        if best is None or t["index"] < best["index"]:
            # Prefer the *lowest* index — weight constants are stored
            # early in the graph, while activation tensors appear later.
            best = t
            best_transposed = transposed

    if best is None:
        return None

    weights = interpreter.get_tensor(best["index"])
    if best_transposed:
        weights = weights.T  # → (features, num_classes)
    return weights


def _compute_heatmap_tflite(
    interpreter: tf.lite.Interpreter,
    img_array: np.ndarray,
    predicted_idx: int,
    num_classes: int,
) -> np.ndarray | None:
    """
    CAM for TFLite models.

    For ``GlobalAveragePooling → Dense`` architectures::

        cam[h, w] = Σ_k  W[k, predicted_class] · feature_map[h, w, k]

    This is mathematically equivalent to Grad-CAM.

    Returns a 2-D array (H, W) with values in [0, 1], or ``None``.
    """
    # ── Run inference to populate intermediate tensors ──
    input_details = interpreter.get_input_details()
    input_info = input_details[0]
    image = img_array.copy()

    if input_info["dtype"] == np.uint8:
        scale, zero_point = input_info["quantization"]
        image = (image / scale + zero_point).astype(np.uint8)
    else:
        image = image.astype(np.float32)

    interpreter.set_tensor(input_info["index"], image)
    interpreter.invoke()

    # ── Get the last conv feature map ──
    conv_info = _find_last_conv_tensor_tflite(interpreter)
    if conv_info is None:
        logger.warning("No spatial feature-map tensor found in TFLite model.")
        return None

    conv_output = interpreter.get_tensor(conv_info["index"])[0]  # (H, W, C)
    logger.info(
        "TFLite CAM: tensor '%s' shape=%s",
        conv_info["name"], conv_output.shape,
    )

    # ── Get Dense weights ──
    dense_weights = _find_dense_weights_tflite(interpreter, num_classes)
    if dense_weights is None:
        logger.warning("Dense weight matrix not found in TFLite model.")
        return None

    # dense_weights: (C, num_classes) → class_weights: (C,)
    class_weights = dense_weights[:, predicted_idx]

    # ── Compute CAM ──
    heatmap = conv_output.astype(np.float32) @ class_weights.astype(np.float32)  # (H, W)

    # ReLU & normalise
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


# ══════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════

def generate_gradcam(
    model_id: str,
    cached_model: tf.keras.Model | tf.lite.Interpreter | None,
    original_image: Image.Image,
    img_array: np.ndarray,
    predicted_idx: int,
) -> Path | None:
    """
    Generate a Grad-CAM / CAM heatmap overlay and save it to ``GRADCAM_DIR``.

    * Keras model  → true Grad-CAM (``GradientTape``).
    * TFLite model → CAM (Dense weights × conv feature map).

    Parameters
    ----------
    model_id       : registry key for the model.
    cached_model   : the cached model object (Keras *or* TFLite interpreter).
    original_image : original PIL image (before pre-processing).
    img_array      : pre-processed numpy array (1, H, W, 3).
    predicted_idx  : index of the predicted class.

    Returns
    -------
    Path to the saved heatmap image, or ``None`` if generation fails.
    """
    try:
        heatmap: np.ndarray | None = None

        if isinstance(cached_model, tf.keras.Model):
            # ── Keras → true Grad-CAM ──
            conv_layer_name = _find_last_conv_layer_keras(cached_model)
            if conv_layer_name is None:
                logger.warning("No conv layer found – skipping Grad-CAM.")
                return None
            logger.info(
                "Generating Grad-CAM (Keras) using layer '%s' …",
                conv_layer_name,
            )
            heatmap = _compute_heatmap_keras(
                cached_model, img_array, predicted_idx, conv_layer_name,
            )

        elif isinstance(cached_model, tf.lite.Interpreter):
            # ── TFLite → CAM ──
            logger.info("Generating CAM (TFLite) for model '%s' …", model_id)
            heatmap = _compute_heatmap_tflite(
                cached_model, img_array, predicted_idx, len(CLASS_NAMES),
            )

        else:
            logger.warning(
                "Unsupported model type for heatmap: %s", type(cached_model),
            )
            return None

        if heatmap is None:
            return None

        overlay = _overlay_heatmap(original_image, heatmap)

        filename = f"gradcam_{uuid.uuid4()}.png"
        output_path = GRADCAM_DIR / filename
        overlay.save(str(output_path), format="PNG")

        logger.info("✅ Heatmap saved: %s", output_path.name)
        return output_path

    except Exception as exc:
        logger.error("Heatmap generation failed: %s", exc, exc_info=True)
        return None
