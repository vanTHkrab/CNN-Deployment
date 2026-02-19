"""Service layer – Grad-CAM (Gradient-weighted Class Activation Mapping).

Generates heatmap overlays that highlight the image regions most
influential to the model's prediction.  Works with **Keras** models
directly and with **TFLite** models by loading the corresponding Keras
weights on-the-fly.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.app.config import GRADCAM_DIR, IMAGE_SIZE, MODEL_REGISTRY

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _find_last_conv_layer(model: tf.keras.Model) -> str | None:
    """
    Recursively walk all layers (including nested sub-models) in reverse
    and return the **name** of the last convolutional layer.

    Supports Conv2D, DepthwiseConv2D, and SeparableConv2D — covering
    architectures like EfficientNet, MobileNet, ConvNeXt, DenseNet, NASNet.
    """
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )

    def _collect_layers(layer: tf.keras.layers.Layer) -> list[tf.keras.layers.Layer]:
        """Flatten all layers, recursing into sub-models / nested Models."""
        layers: list[tf.keras.layers.Layer] = []
        if hasattr(layer, "layers"):  # It's a Model or a wrapper
            for sub in layer.layers:
                layers.extend(_collect_layers(sub))
        else:
            layers.append(layer)
        return layers

    all_layers = _collect_layers(model)

    for layer in reversed(all_layers):
        if isinstance(layer, conv_types):
            return layer.name
    return None


def _get_keras_model_for_gradcam(
    model_id: str,
    cached_model: tf.keras.Model | None,
) -> tf.keras.Model | None:
    """
    Return a Keras model suitable for Grad-CAM.

    * If *cached_model* is already a Keras model → return it directly.
    * If the cached model is a TFLite interpreter → try loading the
      corresponding ``.keras`` file from the registry (without caching it
      long-term — it's only used for this one heatmap).
    * Returns ``None`` when no Keras weights are available.
    """
    if cached_model is not None and isinstance(cached_model, tf.keras.Model):
        return cached_model

    # TFLite path — try to find the .keras file
    meta = MODEL_REGISTRY.get(model_id, {})
    keras_path = meta.get("keras_path")
    if keras_path and Path(keras_path).exists():
        logger.info("Loading Keras model for Grad-CAM (model_id=%s) …", model_id)
        return tf.keras.models.load_model(str(keras_path))

    logger.warning("No Keras weights available for Grad-CAM (model_id=%s).", model_id)
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


def _compute_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    predicted_idx: int,
    conv_layer_name: str,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap — **Keras 3 compatible**.

    For finetuned models the architecture is typically::

        outer_model
         ├── backbone (a tf.keras.Model, e.g. EfficientNetV2B0)
         │    └── … → conv_layer → … → backbone_output
         ├── GlobalAveragePooling2D
         ├── Dropout (optional)
         └── Dense (softmax)

    Strategy:
      1. Build an extractor from the **backbone's own graph**
         (``backbone.input → [conv_layer.output, backbone.output]``).
         This is valid because both tensors belong to the same graph.
      2. Call the extractor inside ``GradientTape`` to get
         ``conv_output`` (watched) and ``backbone_output``.
      3. Eagerly replay the head layers on ``backbone_output``
         to get ``predictions``.
      4. Compute gradients of the target score w.r.t. ``conv_output``.

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
        # ── Nested / finetuned architecture ──
        # Build extractor *within* the backbone's own graph — this is
        # the key difference: both target_layer.output and sub_model.output
        # share the same computational graph so Keras 3 accepts it.
        # NOTE: In Keras 3, sub_model.output can be a list even for
        # single-output models.  Always unwrap to a single tensor.
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

            # Replay the head layers eagerly
            x = backbone_out
            for lyr in model.layers[head_start_idx:]:
                x = lyr(x)
            predictions = x
            target_score = predictions[:, predicted_idx]

        grads = tape.gradient(target_score, conv_output)

    else:
        # ── Flat model (conv layer is a top-level child) ──
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
        raise RuntimeError(
            "Gradients could not be computed. The conv layer output "
            "may be disconnected from the predictions."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # (filters,)

    conv_out = conv_output[0]                                    # (h, w, filters)
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]           # (h, w, 1)
    heatmap = tf.squeeze(heatmap).numpy()

    # ReLU & normalise
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def _overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Overlay the Grad-CAM *heatmap* on the *original_image*.

    Returns a new PIL Image (RGB) of the same size as *original_image*.
    """
    import matplotlib.cm as cm

    # Resize heatmap to match original image dimensions
    w, h = original_image.size
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Apply colourmap  (jet is the classic Grad-CAM palette)
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]      # drop alpha
    heatmap_colored = np.uint8(heatmap_colored * 255)

    # Blend
    heatmap_img = Image.fromarray(heatmap_colored)
    blended = Image.blend(original_image.convert("RGB"), heatmap_img, alpha=alpha)
    return blended


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def generate_gradcam(
    model_id: str,
    cached_model: tf.keras.Model | tf.lite.Interpreter | None,
    original_image: Image.Image,
    img_array: np.ndarray,
    predicted_idx: int,
) -> Path | None:
    """
    Generate a Grad-CAM heatmap overlay and save it to ``GRADCAM_DIR``.

    Parameters
    ----------
    model_id       : registry key for the model.
    cached_model   : the cached model object (Keras or TFLite interpreter).
    original_image : original PIL image (before pre-processing).
    img_array      : pre-processed numpy array (1, H, W, 3).
    predicted_idx  : index of the predicted class.

    Returns
    -------
    Path to the saved heatmap image, or ``None`` if generation fails.
    """
    try:
        # Resolve a Keras model (loads .keras on-the-fly for TFLite)
        keras_model = _get_keras_model_for_gradcam(model_id, cached_model)
        if keras_model is None:
            logger.warning("Cannot generate Grad-CAM – no Keras model available.")
            return None

        conv_layer_name = _find_last_conv_layer(keras_model)
        if conv_layer_name is None:
            logger.warning("No convolutional layer found – skipping Grad-CAM.")
            return None

        logger.info("Generating Grad-CAM using layer '%s' …", conv_layer_name)

        heatmap = _compute_heatmap(keras_model, img_array, predicted_idx, conv_layer_name)
        overlay = _overlay_heatmap(original_image, heatmap)

        # Save to grad-cam directory with unique name
        filename = f"gradcam_{uuid.uuid4()}.png"
        output_path = GRADCAM_DIR / filename
        overlay.save(str(output_path), format="PNG")

        logger.info("✅ Grad-CAM saved: %s", output_path.name)
        return output_path

    except Exception as exc:
        logger.error("Grad-CAM generation failed: %s", exc, exc_info=True)
        return None
