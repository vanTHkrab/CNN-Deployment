from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# ──────────────────────────────────────────────
# Settings (from environment variables / .env)
# ──────────────────────────────────────────────
class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # CORS settings
    cors_origins: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000"
    cors_allow_credentials: bool = True
    cors_allow_methods: str = "GET,POST,PUT,DELETE,OPTIONS"
    cors_allow_headers: str = "*"

    # Upload settings
    max_upload_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_extensions: str = ".jpg,.jpeg,.png,.webp"

    # Application settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def cors_methods_list(self) -> list[str]:
        """Parse CORS methods from comma-separated string."""
        if self.cors_allow_methods == "*":
            return ["*"]
        return [method.strip() for method in self.cors_allow_methods.split(",")]

    @property
    def cors_headers_list(self) -> list[str]:
        """Parse CORS headers from comma-separated string."""
        if self.cors_allow_headers == "*":
            return ["*"]
        return [header.strip() for header in self.cors_allow_headers.split(",")]

    @property
    def allowed_extensions_set(self) -> set[str]:
        """Parse allowed extensions from comma-separated string."""
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",")}


# Global settings instance
settings = Settings()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # src/
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"

# Create upload directory if it doesn't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Class labels (alphabetical order – must match training)
# ──────────────────────────────────────────────
CLASS_NAMES: list[str] = [
    "Allwell",
    "Lifebox",
    "Omron",
    "Sinocare",
    "Yuwell",
]

# ──────────────────────────────────────────────
# Model registry
#   key   → unique model id (used as query param)
#   value → dict with:
#       - name: display name
#       - keras_path: path to .keras file (preferred)
#       - tflite_path: path to .tflite file (fallback)
# ──────────────────────────────────────────────
MODEL_REGISTRY: dict[str, dict] = {
    "efficientnetv2b0": {
        "name": "EfficientNetV2B0",
        "keras_path": MODELS_DIR / "EfficientNetV2B0" / "best_EfficientNetV2B0_finetuned.keras",
        "tflite_path": MODELS_DIR / "EfficientNetV2B0" / "best_EfficientNetV2B0_finetuned.tflite",
    },
    "mobilenetv3large": {
        "name": "MobileNetV3Large",
        "keras_path": MODELS_DIR / "MobileNetV3Large" / "best_MobileNetV3Large_finetuned.keras",
        "tflite_path": MODELS_DIR / "MobileNetV3Large" / "best_MobileNetV3Large_finetuned.tflite",
    },
    "densenet121": {
        "name": "DenseNet121",
        "keras_path": MODELS_DIR / "DenseNet121" / "best_DenseNet121_finetuned.keras",
        "tflite_path": MODELS_DIR / "DenseNet121" / "best_DenseNet121_finetuned.tflite",
    },
    "convnexttiny": {
        "name": "ConvNeXtTiny",
        "keras_path": MODELS_DIR / "ConvNeXtTiny" / "best_ConvNeXtTiny_finetuned.keras",
        "tflite_path": MODELS_DIR / "ConvNeXtTiny" / "best_ConvNeXtTiny_finetuned.tflite",
    },
    "nasnetmobile": {
        "name": "NASNetMobile",
        "keras_path": MODELS_DIR / "NASNetMobile" / "best_NASNetMobile_finetuned.keras",
        "tflite_path": MODELS_DIR / "NASNetMobile" / "best_NASNetMobile_finetuned.tflite",
    },
}

# ──────────────────────────────────────────────
# Image pre-processing defaults
# ──────────────────────────────────────────────
IMAGE_SIZE: tuple[int, int] = (224, 224)
