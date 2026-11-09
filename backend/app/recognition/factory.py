"""Factory registry for creating sign and word recognizers."""

from backend.app.core.exceptions import UnsupportedLanguageError
from backend.app.recognition.base import SignRecognizer, WordRecognizer


class RecognizerFactory:
    """Factory and registry for instantiating language-specific recognizer strategies."""

    _registry: dict[str, type[SignRecognizer]] = {}
    _word_registry: dict[str, type[WordRecognizer]] = {}

    @classmethod
    def register(cls, language: str, impl: type[SignRecognizer]) -> None:
        """Register a single-frame SignRecognizer implementation for a language code."""
        cls._registry[language.upper()] = impl

    @classmethod
    def create(cls, language: str) -> SignRecognizer:
        """Create and return a SignRecognizer for the specified language.

        Args:
            language: Language code (e.g. 'ASL', 'ISL').

        Returns:
            An instance of SignRecognizer.

        Raises:
            UnsupportedLanguageError: If the language code is not registered.
        """
        lang_key = language.upper()
        if lang_key not in cls._registry:
            raise UnsupportedLanguageError(f"Sign language '{language}' is not supported")
        return cls._registry[lang_key]()

    @classmethod
    def register_word(cls, language: str, impl: type[WordRecognizer]) -> None:
        """Register a sequence WordRecognizer implementation for a language code."""
        cls._word_registry[language.upper()] = impl

    @classmethod
    def create_word(cls, language: str) -> WordRecognizer:
        """Create and return a WordRecognizer for the specified language.

        Args:
            language: Language code (e.g. 'ASL').

        Returns:
            An instance of WordRecognizer.

        Raises:
            UnsupportedLanguageError: If the language code is not registered.
        """
        lang_key = language.upper()
        if lang_key not in cls._word_registry:
            raise UnsupportedLanguageError(f"Word recognition for '{language}' is not supported")
        return cls._word_registry[lang_key]()

    @classmethod
    def get_supported_languages(cls) -> list[str]:
        """Return sorted list of supported letter recognition languages."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_supported_word_languages(cls) -> list[str]:
        """Return sorted list of supported word recognition languages."""
        return sorted(cls._word_registry.keys())
