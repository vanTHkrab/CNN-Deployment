"""CNN Deployment – FastAPI application entry-point."""

from contextlib import asynccontextmanager
import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from src.app.config import UPLOAD_DIR, settings
from src.app.router import health, models, predict, upload
from src.app.services.model_service import clear_models, load_all_models

# Configure logging from settings
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Lifespan: load models on startup, release on shutdown
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("🚀 Loading all models …")
    load_all_models()
    logger.info("✅ All models loaded.")
    yield
    logger.info("🛑 Shutting down – clearing model cache …")
    clear_models()


# ──────────────────────────────────────────────
# Application factory
# ──────────────────────────────────────────────
app = FastAPI(
    title="CNN Deployment API",
    description="Predict blood-pressure monitor brands from images.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware (configured from environment variables) ──
app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_methods_list,
    allow_headers=settings.cors_headers_list,
)

logger.info("CORS configured with origins: %s", settings.cors_origins_list)

# ── register routers ──
app.get('/')(lambda: {"message": "Welcome to the CNN Deployment API! Visit /docs for API documentation."})
app.include_router(health.router)
app.include_router(models.router)
app.include_router(upload.router)
app.include_router(predict.router)

# ── serve uploaded images statically ──
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")