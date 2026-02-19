"""Compatibility shim – TFLite runtime import.

In **production** only ``tflite-runtime`` is installed (~5 MB).
In **development** the full ``tensorflow`` package is available.

This module exposes ``Interpreter`` from whichever package is present
so the rest of the code-base never imports ``tensorflow`` directly for
TFLite work.
"""

from __future__ import annotations

try:
    # Production: lightweight tflite-runtime package
    from tflite_runtime.interpreter import Interpreter  # type: ignore[import-untyped]
except ImportError:
    # Development: full TensorFlow
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore[import-untyped]

__all__ = ["Interpreter"]
