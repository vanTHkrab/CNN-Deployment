"""Router – image prediction with Grad-CAM."""

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from src.app.config import GRADCAM_DIR, MODEL_REGISTRY
from src.app.schemas.predict import PredictionRequest, PredictionResponse
from src.app.services.gradcam_service import generate_gradcam
from src.app.services.model_service import (
    fetch_image,
    get_model,
    predict,
    preprocess_image,
)
from src.app.services.storage_service import enforce_storage_limit

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(body: PredictionRequest, request: Request) -> PredictionResponse:
    """
    Predict the class of a blood-pressure monitor image.

    Parameters
    ----------
    body.image_url : str   – public URL of the image to classify.
    body.model_id  : str   – id returned by ``GET /get-models``.

    Returns a prediction **plus** a Grad-CAM heatmap URL.
    """
    # ── validate model id ──
    if body.model_id not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{body.model_id}' not found. "
                   f"Use GET /get-models for available models.",
        )

    # ── fetch & preprocess ──
    try:
        image = await fetch_image(str(body.image_url))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not fetch image: {exc}",
        )

    processed = preprocess_image(image)

    # ── predict ──
    result = predict(body.model_id, processed)

    # ── Grad-CAM (works for Keras directly; loads .keras for TFLite) ──
    gradcam_url: str | None = None

    model = get_model(body.model_id)
    predicted_idx = int(np.argmax([p["confidence"] for p in result["probabilities"]]))

    gradcam_path = generate_gradcam(
        model_id=body.model_id,
        cached_model=model,
        original_image=image,
        img_array=processed,
        predicted_idx=predicted_idx,
    )

    if gradcam_path is not None:
        # Enforce storage limit on grad-cam folder
        enforce_storage_limit(GRADCAM_DIR)

        base_url = str(request.base_url).rstrip("/")
        gradcam_url = f"{base_url}/grad-cam/{gradcam_path.name}"

    return PredictionResponse(
        model_id=body.model_id,
        model_name=MODEL_REGISTRY[body.model_id]["name"],
        image_url=str(body.image_url),
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        gradcam_url=gradcam_url,
    )
