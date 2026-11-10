"""Unit tests for WordSignService threshold, margin, and session integration."""

from backend.app.core.events import EventBus
from backend.app.domain.sequence import LandmarkFrame, LandmarkWindow
from backend.app.recognition.base import FakeWordRecognizer
from backend.app.services.sessions import SessionManager
from backend.app.services.word_sign import WordSignService


def _create_mock_window(n_frames: int = 15) -> LandmarkWindow:
    """Generate a valid LandmarkWindow for testing."""
    pose = tuple((0.5, 0.5) for _ in range(9))
    hand = tuple((0.5, 0.5) for _ in range(21))
    frames = [
        LandmarkFrame(pose=pose, left_hand=hand, right_hand=None)
        for _ in range(n_frames)
    ]
    return LandmarkWindow(frames)


def test_word_service_accepts_word_meeting_threshold_and_margin() -> None:
    bus = EventBus()
    events_received: list[dict] = []
    bus.subscribe("word_recognised", lambda payload: events_received.append(payload))

    sessions = SessionManager()
    service = WordSignService(
        sessions=sessions,
        bus=bus,
        confidence_threshold=0.50,
        margin_threshold=0.10,
    )

    # Top-1: 0.85, Top-2: 0.10 -> diff 0.75 >= 0.10, conf >= 0.50
    fake = FakeWordRecognizer(
        default_predictions=[("hello", 0.85), ("thanks", 0.10), ("yes", 0.05)]
    )
    service.set_recognizer("ASL", fake)

    window = _create_mock_window()
    result = service.recognize_window(session_id="test-session-1", language="ASL", window=window)

    assert result.state == "COMPLETE"
    assert result.accepted_word == "hello"
    assert result.sentence == "HELLO"
    assert result.message is None
    assert len(events_received) == 1
    assert events_received[0]["word"] == "hello"


def test_word_service_rejects_when_below_confidence_threshold() -> None:
    sessions = SessionManager()
    service = WordSignService(
        sessions=sessions,
        confidence_threshold=0.50,
        margin_threshold=0.10,
    )

    # Top-1: 0.45 < 0.50
    fake = FakeWordRecognizer(
        default_predictions=[("hello", 0.45), ("thanks", 0.10), ("yes", 0.05)]
    )
    service.set_recognizer("ASL", fake)

    window = _create_mock_window()
    result = service.recognize_window(session_id="test-session-2", language="ASL", window=window)

    assert result.state == "COMPLETE"
    assert result.accepted_word is None
    assert result.message == "No confident match, please try again"


def test_word_service_rejects_when_below_margin() -> None:
    sessions = SessionManager()
    service = WordSignService(
        sessions=sessions,
        confidence_threshold=0.50,
        margin_threshold=0.10,
    )

    # Top-1: 0.55 >= 0.50, but Top-2: 0.50 -> diff = 0.05 < 0.10
    fake = FakeWordRecognizer(
        default_predictions=[("hello", 0.55), ("thanks", 0.50)]
    )
    service.set_recognizer("ASL", fake)

    window = _create_mock_window()
    result = service.recognize_window(session_id="test-session-3", language="ASL", window=window)

    assert result.state == "COMPLETE"
    assert result.accepted_word is None
    assert result.message == "No confident match, please try again"


def test_word_service_streaming_process_frame_lifecycle() -> None:
    sessions = SessionManager()
    service = WordSignService(
        sessions=sessions,
        confidence_threshold=0.50,
        margin_threshold=0.10,
    )
    fake = FakeWordRecognizer(default_predictions=[("thankyou", 0.90), ("yes", 0.05)])
    service.set_recognizer("ASL", fake)

    pose = tuple((0.5, 0.5) for _ in range(9))

    # Frame 0: idle -> returns WAITING
    empty_frame = LandmarkFrame(pose=pose, left_hand=None, right_hand=None)
    res0 = service.process_frame("stream-session", "ASL", empty_frame)
    assert res0.state == "WAITING"

    # Moving frames -> transitions to SIGNING
    for i in range(15):
        h = tuple((0.1 + (i * 0.02), 0.1) for _ in range(21))
        f = LandmarkFrame(pose=pose, left_hand=h, right_hand=None)
        res = service.process_frame("stream-session", "ASL", f)
        assert res.state == "SIGNING"

    # Idle frames to finish sign (end_idle_frames is 8 by default)
    last_res = None
    for _ in range(8):
        last_res = service.process_frame("stream-session", "ASL", empty_frame)

    assert last_res is not None
    assert last_res.state == "COMPLETE"
    assert last_res.accepted_word == "thankyou"
    assert last_res.sentence == "THANKYOU"
