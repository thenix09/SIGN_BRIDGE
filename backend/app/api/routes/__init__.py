"""API endpoint route definitions."""

from backend.app.api.routes import (
    health,
    history,
    languages,
    recognize,
    samples,
    text_to_sign,
    words,
)

__all__ = ["health", "history", "languages", "recognize", "samples", "text_to_sign", "words"]
