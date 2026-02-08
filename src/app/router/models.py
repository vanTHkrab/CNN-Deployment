"""Router – model catalogue."""

from fastapi import APIRouter

from src.app.config import MODEL_REGISTRY
from src.app.schemas.model import ModelInfo, ModelsResponse

router = APIRouter(tags=["Models"])


@router.get("/get-models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    """Return every available model with its id and display name."""
    models = [
        ModelInfo(id=model_id, name=meta["name"])
        for model_id, meta in MODEL_REGISTRY.items()
    ]
    return ModelsResponse(models=models)
