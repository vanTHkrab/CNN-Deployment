"""Service layer – storage management for uploaded and generated files."""

from __future__ import annotations

import logging
from pathlib import Path

from src.app.config import settings

logger = logging.getLogger(__name__)


def enforce_storage_limit(directory: Path) -> None:
    """
    Keep at most ``settings.max_storage_files`` image files in *directory*.

    When the limit is exceeded the **oldest** files (by modification time)
    are deleted until the count is back within the limit.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    files = sorted(
        (f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in image_extensions),
        key=lambda f: f.stat().st_mtime,
    )

    max_files = settings.max_storage_files
    if len(files) <= max_files:
        return

    files_to_delete = files[: len(files) - max_files]
    for file in files_to_delete:
        try:
            file.unlink()
            logger.info("🗑️  Deleted old file: %s", file.name)
        except OSError as exc:
            logger.warning("Failed to delete %s: %s", file.name, exc)
