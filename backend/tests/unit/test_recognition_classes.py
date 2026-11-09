"""Unit tests for recognition subsystem: StabilityFilter, WordBuilder, SuggestionEngine, and RecognizerFactory."""

import pytest

from backend.app.core.exceptions import ModelNotAvailableError, UnsupportedLanguageError
from backend.app.domain.gesture import Gesture
from backend.app.recognition.asl_recognizer import ASLRecognizer
from backend.app.recognition.factory import RecognizerFactory
from backend.app.recognition.isl_recognizer import ISLRecognizer
from backend.app.recognition.stability import RecognizerState, StabilityFilter
from backend.app.recognition.words import SuggestionEngine, WordBuilder


def test_stability_filter_lifecycle() -> None:
    """Verify stability filter states: IDLE -> DETECTING -> STABILISING -> CONFIRMED."""
    filter_ = StabilityFilter(required_frames=3, cooldown_frames=3)

    assert filter_.state == RecognizerState.IDLE

    # Frame 1: Detecting
    state, confirmed = filter_.update(Gesture(label="A", confidence=0.9))
    assert state == RecognizerState.DETECTING
    assert confirmed is None

    # Frame 2: Stabilising
    state, confirmed = filter_.update(Gesture(label="A", confidence=0.9))
    assert state == RecognizerState.STABILISING
    assert confirmed is None

    # Frame 3: Confirmed
    state, confirmed = filter_.update(Gesture(label="A", confidence=0.9))
    assert state == RecognizerState.CONFIRMED
    assert confirmed == "A"

    # Frame 4: Emitted exactly once; next frame is stabilising
    state, confirmed = filter_.update(Gesture(label="A", confidence=0.9))
    assert state == RecognizerState.STABILISING
    assert confirmed is None


def test_stability_filter_resets_on_hand_loss() -> None:
    """Verify losing hand resets filter to IDLE."""
    filter_ = StabilityFilter(required_frames=5)
    filter_.update(Gesture(label="B", confidence=0.8))
    assert filter_.state == RecognizerState.DETECTING

    state, confirmed = filter_.update(None)
    assert state == RecognizerState.IDLE
    assert confirmed is None


def test_word_builder_operations() -> None:
    """Verify add_letter, backspace, commit_word, and clear."""
    builder = WordBuilder()
    assert builder.current_word == ""
    assert builder.sentence == ""

    builder.add_letter("h")
    builder.add_letter("e")
    builder.add_letter("l")
    builder.add_letter("l")
    builder.add_letter("o")
    assert builder.current_word == "HELLO"
    assert builder.sentence == "HELLO"

    # Backspace
    builder.backspace()
    assert builder.current_word == "HELL"

    # Commit with suggestion replacement
    builder.commit_word("HELLO")
    assert builder.current_word == ""
    assert builder.sentence == "HELLO"

    # Add second word
    builder.add_letter("W")
    builder.add_letter("O")
    builder.add_letter("R")
    builder.add_letter("L")
    builder.add_letter("D")
    builder.commit_word()
    assert builder.sentence == "HELLO WORLD"

    builder.clear()
    assert builder.current_word == ""
    assert builder.sentence == ""


def test_suggestion_engine() -> None:
    """Verify Trie prefix search and limits."""
    words = ["APPLE", "APPLICATION", "APPLY", "BANANA", "CAT"]
    engine = SuggestionEngine(words)

    # Prefix match
    res = engine.suggest("APP", limit=2)
    assert len(res) == 2
    assert "APPLE" in res

    # Limit 3
    res_all = engine.suggest("APP", limit=5)
    assert len(res_all) == 3
    assert set(res_all) == {"APPLE", "APPLICATION", "APPLY"}

    # No match
    assert engine.suggest("DOG") == []
    # Empty prefix
    assert engine.suggest("") == []


def test_recognizer_factory() -> None:
    """Verify factory registration, creation, and language errors."""
    asl = RecognizerFactory.create("ASL")
    assert isinstance(asl, ASLRecognizer)
    assert asl.language == "ASL"

    # Unknown language raises UnsupportedLanguageError
    with pytest.raises(UnsupportedLanguageError):
        RecognizerFactory.create("FRENCH_SIGN")

    # ISL stub creates but raises ModelNotAvailableError on load/recognize
    isl = RecognizerFactory.create("ISL")
    assert isinstance(isl, ISLRecognizer)
    with pytest.raises(ModelNotAvailableError):
        isl.load()
