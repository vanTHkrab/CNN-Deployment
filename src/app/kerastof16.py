"""
Script to convert all Keras models to TFLite format with FP16 optimization.

This script converts each .keras model to a .tflite file in the same directory.
Use this to reduce model file sizes for deployment.

Usage:
    python -m src.app.kerastof16
"""

import tensorflow as tf
from pathlib import Path

# Get the models directory
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Model directories
MODEL_FOLDERS = [
    "ConvNeXtTiny",
    # "DenseNet121",
    # "EfficientNetV2B0",
    # "MobileNetV3Large",
    # "NASNetMobile",
]

def convert_keras_to_tflite(keras_path: Path, tflite_path: Path) -> None:
    """Convert a Keras model to TFLite with FP16 optimization."""
    print(f"Loading {keras_path.name}...")
    model = tf.keras.models.load_model(str(keras_path))
    
    print(f"Converting to TFLite (FP16)...")
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]

    print(f"Converting to TFLite (INT8)...")
    def representative_dataset():
        for _ in range(100):
            yield [tf.random.normal([1, 224, 224, 3])]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    print(f"Saving to {tflite_path.name}...")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    # Get file sizes
    keras_size = keras_path.stat().st_size / (1024 * 1024)  # MB
    tflite_size = tflite_path.stat().st_size / (1024 * 1024)  # MB
    reduction = ((keras_size - tflite_size) / keras_size) * 100
    
    print(f"✅ Converted: {keras_path.name}")
    print(f"   Keras:  {keras_size:.2f} MB")
    print(f"   TFLite: {tflite_size:.2f} MB")
    print(f"   Reduction: {reduction:.1f}%\n")


def main():
    """Convert all Keras models to TFLite format."""
    print("="*60)
    print("Converting Keras models to TFLite (FP16)")
    print("="*60 + "\n")
    
    total_converted = 0
    
    for folder in MODEL_FOLDERS:
        model_dir = MODELS_DIR / folder
        if not model_dir.exists():
            print(f"⚠️  Skipping {folder} (directory not found)")
            continue
        
        # Convert both base and finetuned models
        for keras_file in model_dir.glob("*.keras"):
            tflite_file = keras_file.with_suffix(".tflite")
            
            try:
                convert_keras_to_tflite(keras_file, tflite_file)
                total_converted += 1
            except Exception as e:
                print(f"❌ Error converting {keras_file.name}: {e}\n")
    
    print("="*60)
    print(f"✅ Conversion complete! {total_converted} models converted.")
    print("="*60)


if __name__ == "__main__":
    main()

