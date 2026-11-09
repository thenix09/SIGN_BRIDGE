"""FastAPI dependency providers."""

from functools import lru_cache

from backend.app.core.config import Settings, get_settings
from backend.app.core.events import default_event_bus
from backend.app.recognition.factory import RecognizerFactory
from backend.app.recognition.words import SuggestionEngine
from backend.app.services.sessions import SessionManager, default_session_manager
from backend.app.services.sign_to_text import SignToTextService


def get_app_settings() -> Settings:
    """Dependency provider returning current application settings."""
    return get_settings()


def get_session_manager() -> SessionManager:
    """Dependency provider returning the SessionManager singleton."""
    return default_session_manager


@lru_cache
def get_suggestion_engine() -> SuggestionEngine:
    """Dependency provider returning the cached SuggestionEngine instance."""
    settings = get_settings()
    return SuggestionEngine(settings.words_path)


def get_sign_to_text_service() -> SignToTextService:
    """Dependency provider returning the SignToTextService."""
    return SignToTextService(
        factory=RecognizerFactory,
        sessions=get_session_manager(),
        suggestions=get_suggestion_engine(),
        bus=default_event_bus,
    )
