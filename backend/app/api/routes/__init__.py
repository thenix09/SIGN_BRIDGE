"""API endpoint route definitions."""

from backend.app.api.routes import health, languages, recognize, text_to_sign, words

__all__ = ["health", "languages", "recognize", "text_to_sign", "words"]
