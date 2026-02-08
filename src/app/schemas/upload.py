from pydantic import BaseModel


class UploadResponse(BaseModel):
    """Response schema for POST /upload."""
    url: str
    filename: str
    size: int
