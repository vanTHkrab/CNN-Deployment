"""Router – image prediction."""

from fastapi import APIRouter, HTTPException

from src.app.config import MODEL_REGISTRY
from src.app.schemas.predict import PredictionRequest, PredictionResponse
from src.app.services.model_service import (
    fetch_image,
    predict,
    preprocess_image,
)

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(body: PredictionRequest) -> PredictionResponse:
    """
    Predict the class of a blood-pressure monitor image.

    Parameters
    ----------
    body.image_url : str   – public URL of the image to classify.
    body.model_id  : str   – id returned by ``GET /get-models``.
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

    return PredictionResponse(
        model_id=body.model_id,
        model_name=MODEL_REGISTRY[body.model_id]["name"],
        image_url=str(body.image_url),
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )
