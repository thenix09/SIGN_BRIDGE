"""Integration tests for Word recognition and Vocabulary API endpoints."""

from fastapi.testclient import TestClient

from backend.app.api.deps import get_word_sign_service
from backend.app.main import app
from backend.app.recognition.base import FakeWordRecognizer
from backend.app.services.sessions import SessionManager
from backend.app.services.word_sign import WordSignService

# Inject a mock word sign service
_test_sessions = SessionManager()
_test_word_service = WordSignService(sessions=_test_sessions)
_fake_recognizer = FakeWordRecognizer(
    default_predictions=[("hello", 0.90), ("thankyou", 0.05), ("yes", 0.02)]
)
_test_word_service.set_recognizer("ASL", _fake_recognizer)

app.dependency_overrides[get_word_sign_service] = lambda: _test_word_service
client = TestClient(app)


def _sample_pose() -> list[list[float]]:
    return [[0.5, 0.5] for _ in range(9)]


def _sample_hand() -> list[list[float]]:
    return [[0.5, 0.5] for _ in range(21)]


def test_get_vocabulary_endpoint() -> None:
    response = client.get("/api/vocabulary?language=ASL")
    assert response.status_code == 200
    data = response.json()
    assert "size" in data
    assert "status" in data
    assert "words" in data
    assert isinstance(data["words"], list)
    assert data["status"] in ("available", "withheld", "experimental", "unavailable")


def test_recognize_words_post_single_frame() -> None:
    payload = {
        "session_id": "api-words-sess-1",
        "language": "ASL",
        "frame": {
            "pose": _sample_pose(),
            "hands": [_sample_hand()],
        },
    }
    response = client.post("/api/recognize-words", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["state"] in ("WAITING", "SIGNING", "COMPLETE")
    assert "top_k" in data
    assert "sentence" in data


def test_recognize_words_post_window() -> None:
    # Send pre-accumulated window of 15 frames for direct evaluation
    window_frames = [
        {"pose": _sample_pose(), "hands": [_sample_hand()]}
        for _ in range(15)
    ]
    payload = {
        "session_id": "api-words-sess-2",
        "language": "ASL",
        "window": window_frames,
    }
    response = client.post("/api/recognize-words", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "COMPLETE"
    assert data["accepted_word"] == "hello"
    assert len(data["top_k"]) == 3
    assert data["top_k"][0]["label"] == "hello"
    assert data["sentence"] == "HELLO"


def test_websocket_recognize_words_streaming() -> None:
    with client.websocket_connect("/ws/recognize-words") as ws:
        # 1. Send invalid JSON: should return error and keep connection open
        ws.send_text("not json")
        err_reply = ws.receive_json()
        assert "error" in err_reply

        # 2. Send valid frame
        payload = {
            "session_id": "ws-words-sess",
            "language": "ASL",
            "frame": {
                "pose": _sample_pose(),
                "hands": [_sample_hand()],
            },
        }
        ws.send_json(payload)
        reply = ws.receive_json()
        assert "state" in reply
        assert "top_k" in reply
        assert "sentence" in reply
