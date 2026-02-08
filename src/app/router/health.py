"""Router – health check."""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check() -> dict:
    """Liveness / readiness probe."""
    return {"status": "ok"}
