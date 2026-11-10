import pytest

from backend.app.core.exceptions import UnsupportedLanguageError
from backend.app.recognition import ASLWordRecognizer, RecognizerFactory
from backend.app.recognition.base import SignRecognizer, WordRecognizer


def test_asl_word_recognizer_is_registered() -> None:
    assert "ASL" in RecognizerFactory.get_supported_word_languages()


def test_factory_creates_asl_word_recognizer() -> None:
    recognizer = RecognizerFactory.create_word("ASL")

    assert isinstance(recognizer, ASLWordRecognizer)
    assert isinstance(recognizer, WordRecognizer)
    assert recognizer.language == "ASL"


def test_factory_word_language_lookup_is_case_insensitive() -> None:
    recognizer = RecognizerFactory.create_word("asl")

    assert isinstance(recognizer, ASLWordRecognizer)


def test_factory_rejects_unsupported_word_language() -> None:
    with pytest.raises(UnsupportedLanguageError):
        RecognizerFactory.create_word("UNSUPPORTED")


def test_word_and_letter_recognizer_registries_remain_separate() -> None:
    letter_recognizer = RecognizerFactory.create("ASL")
    word_recognizer = RecognizerFactory.create_word("ASL")

    assert isinstance(letter_recognizer, SignRecognizer)
    assert not isinstance(letter_recognizer, WordRecognizer)
    assert isinstance(word_recognizer, WordRecognizer)
