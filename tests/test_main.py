"""Tests for CNN Deployment API endpoints."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.app.config import CLASS_NAMES, MODEL_REGISTRY
from src.app.main import app

client = TestClient(app)


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────
def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ──────────────────────────────────────────────
# GET /get-models
# ──────────────────────────────────────────────
def test_get_models_returns_all() -> None:
    response = client.get("/get-models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == len(MODEL_REGISTRY)


def test_get_models_structure() -> None:
    response = client.get("/get-models")
    data = response.json()
    for model in data["models"]:
        assert "id" in model
        assert "name" in model
        assert model["id"] in MODEL_REGISTRY


# ──────────────────────────────────────────────
# POST /predict
# ──────────────────────────────────────────────
def test_predict_invalid_model_id() -> None:
    response = client.post("/predict", json={
        "image_url": "https://example.com/image.jpg",
        "model_id": "nonexistent_model",
    })
    assert response.status_code == 404


def test_predict_invalid_image_url() -> None:
    response = client.post("/predict", json={
        "image_url": "not-a-url",
        "model_id": "efficientnetv2b0",
    })
    assert response.status_code == 422  # Pydantic validation error


@patch("src.app.router.predict.fetch_image")
@patch("src.app.router.predict.predict")
def test_predict_success(mock_predict: MagicMock, mock_fetch: MagicMock) -> None:
    """End-to-end with mocked model + image download."""
    from PIL import Image

    # mock fetch_image → return a dummy PIL image
    dummy_img = Image.new("RGB", (224, 224))
    mock_fetch.return_value = dummy_img

    # mock predict → return a fake result
    mock_predict.return_value = {
        "predicted_class": CLASS_NAMES[0],
        "confidence": 0.95,
        "probabilities": [
            {"class_name": name, "confidence": round(1.0 / len(CLASS_NAMES), 4)}
            for name in CLASS_NAMES
        ],
    }

    response = client.post("/predict", json={
        "image_url": "https://example.com/image.jpg",
        "model_id": "efficientnetv2b0",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "efficientnetv2b0"
    assert data["model_name"] == "EfficientNetV2B0"
    assert data["predicted_class"] == CLASS_NAMES[0]
    assert "probabilities" in data
    assert len(data["probabilities"]) == len(CLASS_NAMES)


# ──────────────────────────────────────────────
# POST /upload
# ──────────────────────────────────────────────
def test_upload_success() -> None:
    """Test successful image upload."""
    # Create a dummy image in memory
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    response = client.post(
        "/upload",
        files={"file": ("test_image.jpg", img_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "url" in data
    assert "filename" in data
    assert "size" in data
    assert data["filename"] == "test_image.jpg"
    assert data["url"].endswith(".jpg")


def test_upload_invalid_extension() -> None:
    """Test upload with invalid file extension."""
    content = b"fake content"
    response = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
    )
    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"].lower()


def test_upload_no_file() -> None:
    """Test upload without providing a file."""
    response = client.post("/upload")
    assert response.status_code == 422  # Validation error

