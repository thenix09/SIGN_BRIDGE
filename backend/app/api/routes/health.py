"""Health check API route."""

from pathlib import Path

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_app_settings
from backend.app.api.schemas import HealthResponse
from backend.app.core.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Return system health status, model readiness, and version."""
    letter_model_exists = Path(settings.model_path).is_file()
    word_model_exists = Path(settings.word_model_path).is_file()

    return HealthResponse(
        status="ok",
        model_loaded=letter_model_exists,
        word_model_loaded=word_model_exists,
        version="1.0.0",
    )
