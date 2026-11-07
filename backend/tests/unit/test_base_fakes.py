"""Unit tests for recognition and rendering abstract base classes and fakes."""

import pytest

from backend.app.domain.gesture import Gesture
from backend.app.recognition.base import (
    FakeRecognizer,
    FakeWordRecognizer,
    SignRecognizer,
    WordRecognizer,
)
from backend.app.rendering.base import (
    FakeSignRenderer,
    SignFrame,
    SignRenderer,
)


def test_gesture_dataclass() -> None:
    """Verify Gesture properties and immutability."""
    g = Gesture(label="A", confidence=0.98)
    assert g.label == "A"
    assert g.confidence == 0.98
    with pytest.raises(Exception):
        g.confidence = 0.5  # type: ignore[misc]


def test_sign_frame_dataclass() -> None:
    """Verify SignFrame properties and immutability."""
    frame = SignFrame(
        label="H",
        kind="letter",
        pose=[],
        left_hand=None,
        right_hand=[[0.0, 0.0]],
        duration_ms=300,
    )
    assert frame.label == "H"
    assert frame.kind == "letter"
    assert frame.duration_ms == 300
    with pytest.raises(Exception):
        frame.label = "X"  # type: ignore[misc]


def test_fake_recognizer_polymorphism() -> None:
    """Verify FakeRecognizer adheres to SignRecognizer contract."""
    recognizer: SignRecognizer = FakeRecognizer(
        language="ASL", default_label="B", default_confidence=0.88
    )
    assert recognizer.language == "ASL"
    recognizer.load()
    gesture = recognizer.recognize(frame=None)  # type: ignore[arg-type]
    assert gesture is not None
    assert gesture.label == "B"
    assert gesture.confidence == 0.88


def test_fake_recognizer_low_confidence() -> None:
    """Verify FakeRecognizer returns None when confidence is zero or negative."""
    recognizer = FakeRecognizer(default_confidence=0.0)
    assert recognizer.recognize(frame=None) is None  # type: ignore[arg-type]


def test_fake_word_recognizer_polymorphism() -> None:
    """Verify FakeWordRecognizer adheres to WordRecognizer contract."""
    recognizer: WordRecognizer = FakeWordRecognizer(
        language="ASL",
        vocabulary=("hello", "thank you", "yes"),
        default_predictions=[("hello", 0.90), ("thank you", 0.08), ("yes", 0.02)],
    )
    assert recognizer.language == "ASL"
    assert recognizer.vocabulary == ("hello", "thank you", "yes")
    recognizer.load()
    results = recognizer.recognize(window=None, top_k=2)  # type: ignore[arg-type]
    assert len(results) == 2
    assert results[0].label == "hello"
    assert results[0].confidence == 0.90
    assert results[1].label == "thank you"


def test_fake_sign_renderer_polymorphism() -> None:
    """Verify FakeSignRenderer adheres to SignRenderer contract."""
    renderer: SignRenderer = FakeSignRenderer(default_duration_ms=250)
    frames = renderer.render("HI")
    assert len(frames) == 2
    assert frames[0].label == "H"
    assert frames[0].kind == "letter"
    assert frames[0].duration_ms == 250
    assert frames[1].label == "I"
