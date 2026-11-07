"""FastAPI dependency providers."""

from backend.app.core.config import Settings, get_settings


def get_app_settings() -> Settings:
    """Dependency provider returning current application settings."""
    return get_settings()
