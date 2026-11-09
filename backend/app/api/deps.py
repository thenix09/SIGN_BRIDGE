"""FastAPI dependency providers."""

from functools import lru_cache

from backend.app.core.config import Settings, get_settings
from backend.app.core.events import default_event_bus
from backend.app.recognition.factory import RecognizerFactory
from backend.app.recognition.words import SuggestionEngine
from backend.app.rendering.factory import RendererFactory
from backend.app.services.sessions import SessionManager, default_session_manager
from backend.app.services.sign_to_text import SignToTextService
from backend.app.services.text_to_sign import TextToSignService
from backend.app.services.word_sign import WordSignService


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


def get_word_sign_service() -> WordSignService:
    """Dependency provider returning the WordSignService."""
    return WordSignService(
        factory=RecognizerFactory,
        sessions=get_session_manager(),
        bus=default_event_bus,
    )


def get_text_to_sign_service() -> TextToSignService:
    """Dependency provider returning the TextToSignService."""
    return TextToSignService(
        renderer_factory=RendererFactory,
        bus=default_event_bus,
    )
