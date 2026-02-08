from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Single model entry returned by /get-models."""
    id: str
    name: str


class ModelsResponse(BaseModel):
    """Response schema for GET /get-models."""
    models: list[ModelInfo]
