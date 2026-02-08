from pydantic import BaseModel, HttpUrl


class PredictionRequest(BaseModel):
    """Body schema for POST /predict."""
    image_url: HttpUrl
    model_id: str


class PredictionResult(BaseModel):
    """Single class probability."""
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""
    model_id: str
    model_name: str
    image_url: str
    predicted_class: str
    confidence: float
    probabilities: list[PredictionResult]
