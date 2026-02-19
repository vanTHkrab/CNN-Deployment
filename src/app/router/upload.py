"""Router – file upload."""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.app.config import UPLOAD_DIR, settings
from src.app.schemas.upload import UploadResponse
from src.app.services.storage_service import enforce_storage_limit

router = APIRouter(tags=["Upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload_image(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload an image file and return a publicly accessible URL.

    Parameters
    ----------
    file : UploadFile – image file to upload (jpg, jpeg, png, webp).

    Returns
    -------
    UploadResponse with:
        - url      : public URL to access the uploaded image
        - filename : original filename
        - size     : file size in bytes
    """
    # ── validate file extension ──
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions_set:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_ext}' not allowed. "
                   f"Allowed: {', '.join(settings.allowed_extensions_set)}",
        )

    # ── read file content ──
    content = await file.read()
    file_size = len(content)

    if file_size > settings.max_upload_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size} bytes). "
                   f"Maximum size: {settings.max_upload_size} bytes.",
        )

    # ── save with unique name ──
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename

    with open(file_path, "wb") as f:
        f.write(content)

    # ── enforce storage limit ──
    enforce_storage_limit(UPLOAD_DIR)

    # ── construct public URL ──
    # Use request.url to get the base URL (scheme + host + port)
    base_url = str(request.base_url).rstrip("/")
    image_url = f"{base_url}/uploads/{unique_filename}"

    return UploadResponse(
        url=image_url,
        filename=file.filename,
        size=file_size,
    )
