"""Service layer – model loading, image fetching, and prediction.

Supports both **full TensorFlow** (development) and **tflite-runtime**
(production).  When only ``tflite-runtime`` is installed, Keras-related
functionality is automatically disabled, keeping memory under ~50 MB per
model.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Union

import httpx
import numpy as np
from PIL import Image

from src.app.compat import Interpreter
from src.app.config import CLASS_NAMES, IMAGE_SIZE, MODEL_REGISTRY

logger = logging.getLogger(__name__)

# ── Optional: full TensorFlow (only in dev) ──
try:
    import tensorflow as tf  # type: ignore[import-untyped]

    _HAS_TF = True
except ImportError:
    _HAS_TF = False

# ──────────────────────────────────────────────
# In-memory model cache  (model_id → model object)
# ──────────────────────────────────────────────
_model_cache: dict[str, Any] = {}
_model_types: dict[str, str] = {}   # 'keras' | 'tflite'
_tflite_paths: dict[str, str] = {}  # model_id → .tflite file path (for CAM)


# ──────────────────────────────────────────────
# Model life-cycle helpers
# ──────────────────────────────────────────────
def load_all_models() -> None:
    """Validate that every registered model has at least one file on disk.

    Models are **not** loaded into memory at startup.  They are loaded
    lazily on first request via ``get_model()`` to keep the startup
    footprint small (important in 512 MB containers).
    """
    for model_id, meta in MODEL_REGISTRY.items():
        keras_path = meta.get("keras_path")
        tflite_path = meta.get("tflite_path")

        has_keras = _HAS_TF and keras_path and Path(keras_path).exists()
        has_tflite = tflite_path and Path(tflite_path).exists()

        if has_keras or has_tflite:
            logger.info("✅ Model %s found on disk (lazy load).", model_id)
        else:
            logger.error(
                "No model file found for %s. Expected: %s or %s",
                model_id, keras_path, tflite_path,
            )


def load_keras_model(model_id: str, path: Path) -> Any:
    """Load a Keras model and store it in the cache."""
    if not _HAS_TF:
        raise RuntimeError("Full TensorFlow is required for Keras models.")
    logger.info("Loading Keras model %s from %s …", model_id, path)
    model = tf.keras.models.load_model(str(path))
    _model_cache[model_id] = model
    _model_types[model_id] = "keras"
    logger.info("✅ Keras model %s loaded successfully.", model_id)
    return model


def load_tflite_model(model_id: str, path: Path) -> Interpreter:
    """Load a TFLite model and store it in the cache."""
    logger.info("Loading TFLite model %s from %s …", model_id, path)
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    _model_cache[model_id] = interpreter
    _model_types[model_id] = "tflite"
    _tflite_paths[model_id] = str(path)
    logger.info("✅ TFLite model %s loaded successfully.", model_id)
    return interpreter


def get_model(model_id: str) -> Any:
    """Return a cached model, loading it lazily on first access."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    # Lazy load
    meta = MODEL_REGISTRY.get(model_id)
    if meta is None:
        raise KeyError(f"Unknown model_id: {model_id}")

    keras_path = meta.get("keras_path")
    tflite_path = meta.get("tflite_path")

    if _HAS_TF and keras_path and Path(keras_path).exists():
        return load_keras_model(model_id, keras_path)
    if tflite_path and Path(tflite_path).exists():
        return load_tflite_model(model_id, tflite_path)

    raise FileNotFoundError(
        f"No model file on disk for '{model_id}'.",
    )


def get_model_type(model_id: str) -> str:
    """Return the type of model ('keras' or 'tflite')."""
    return _model_types.get(model_id, "unknown")


def get_tflite_path(model_id: str) -> str | None:
    """Return the ``.tflite`` file path for the given model, or ``None``."""
    return _tflite_paths.get(model_id)


def clear_models() -> None:
    """Release all cached models (called at shutdown)."""
    _model_cache.clear()
    _model_types.clear()
    _tflite_paths.clear()
    if _HAS_TF:
        tf.keras.backend.clear_session()
    logger.info("All models cleared from cache.")


# ──────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────
async def fetch_image(image_url: str) -> Image.Image:
    """Download an image from *image_url* and return a PIL Image."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(str(image_url))
        response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalise an image to a batch-ready numpy array."""
    image = image.resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def predict(model_id: str, image: np.ndarray) -> dict:
    model = get_model(model_id)
    model_type = get_model_type(model_id)

    if model_type == "keras":
        if not _HAS_TF:
            raise RuntimeError("Full TensorFlow is required for Keras models.")
        preds: np.ndarray = model.predict(image, verbose=0)
        probs = preds[0]

    elif model_type == "tflite":
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # ---------- INPUT ----------
        input_info = input_details[0]

        if input_info["dtype"] == np.uint8:
            scale, zero_point = input_info["quantization"]
            image = image / scale + zero_point
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.float32)

        model.set_tensor(input_info["index"], image)
        model.invoke()

        # ---------- OUTPUT ----------
        output_info = output_details[0]
        output_data = model.get_tensor(output_info["index"])[0]

        if output_info["dtype"] == np.uint8:
            scale, zero_point = output_info["quantization"]
            output_data = (output_data.astype(np.float32) - zero_point) * scale

        # softmax
        exp = np.exp(output_data - np.max(output_data))
        probs = exp / exp.sum()

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    predicted_idx = int(np.argmax(probs))

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(float(probs[predicted_idx]), 4),
        "probabilities": [
            {"class_name": name, "confidence": round(float(prob), 4)}
            for name, prob in zip(CLASS_NAMES, probs)
        ],
    }

