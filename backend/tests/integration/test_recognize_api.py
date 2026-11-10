"""Integration tests for fingerspelling recognition routes and WebSocket endpoint."""

from fastapi.testclient import TestClient

from backend.app.api.deps import get_sign_to_text_service
from backend.app.core.events import EventBus
from backend.app.main import create_app
from backend.app.recognition.base import FakeRecognizer, SignRecognizer
from backend.app.recognition.factory import RecognizerFactory
from backend.app.recognition.words import SuggestionEngine
from backend.app.services.sessions import SessionManager
from backend.app.services.sign_to_text import SignToTextService


class TestFactory(RecognizerFactory):
    """Test factory returning FakeRecognizer for ASL."""

    @classmethod
    def create(cls, language: str) -> SignRecognizer:
        if language.upper() == "ASL":
            return FakeRecognizer(language="ASL", default_label="A", default_confidence=0.95)
        return super().create(language)


def _build_test_app() -> tuple[TestClient, EventBus]:
    """Construct app with mock SignToTextService and test client."""
    app = create_app()
    bus = EventBus()
    sessions = SessionManager(ttl_seconds=300)
    suggestions = SuggestionEngine(["ALL", "AND", "ARE", "APPLE"])

    service = SignToTextService(
        factory=TestFactory,
        sessions=sessions,
        suggestions=suggestions,
        bus=bus,
    )

    app.dependency_overrides[get_sign_to_text_service] = lambda: service
    client = TestClient(app)
    return client, bus


def test_get_languages_endpoint() -> None:
    """Verify /api/languages lists ASL and ISL with expected structure."""
    client, _ = _build_test_app()
    resp = client.get("/api/languages")
    assert resp.status_code == 200
    data = resp.json()
    assert "languages" in data
    codes = [item["code"] for item in data["languages"]]
    assert "ASL" in codes
    assert "ISL" in codes


def test_recognize_route_valid_and_invalid() -> None:
    """Verify POST /api/recognize with valid 21 points and invalid input."""
    client, _ = _build_test_app()

    # Valid payload
    payload = {
        "session_id": "test-session-1",
        "language": "ASL",
        "handedness": "Right",
        "landmarks": [[float(i) * 0.01, 0.5, 0.2] for i in range(21)],
    }
    resp = client.post("/api/recognize", json=payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["label"] == "A"
    assert res_data["confidence"] == 0.95
    assert res_data["state"] in ["DETECTING", "STABILISING", "CONFIRMED"]

    # Invalid landmark count (only 10 points)
    bad_payload = dict(payload)
    bad_payload["landmarks"] = [[0.1, 0.2, 0.3]] * 10
    resp_bad = client.post("/api/recognize", json=bad_payload)
    assert resp_bad.status_code == 422


def test_session_action_endpoints() -> None:
    """Verify session reset, backspace, and commit endpoints."""
    client, bus = _build_test_app()
    session_id = "test-session-actions"

    # Send 8 identical frames to confirm letter 'A'
    payload = {
        "session_id": session_id,
        "language": "ASL",
        "handedness": "Right",
        "landmarks": [[float(i) * 0.01, 0.5, 0.2] for i in range(21)],
    }
    for _ in range(8):
        client.post("/api/recognize", json=payload)

    # Commit word
    resp_commit = client.post("/api/session/commit", json={"session_id": session_id})
    assert resp_commit.status_code == 200
    assert resp_commit.json()["sentence"] == "A"

    # Backspace
    resp_backspace = client.post("/api/session/backspace", json={"session_id": session_id})
    assert resp_backspace.status_code == 200

    # Reset
    resp_reset = client.post("/api/session/reset", json={"session_id": session_id})
    assert resp_reset.status_code == 200
    assert resp_reset.json()["ok"] is True


def test_websocket_recognize_stream() -> None:
    """Verify WebSocket /ws/recognize streams results and handles bad frames gracefully."""
    client, _ = _build_test_app()

    with client.websocket_connect("/ws/recognize") as ws:
        # 1. Send valid frame
        valid_frame = {
            "session_id": "ws-test-session",
            "language": "ASL",
            "handedness": "Right",
            "landmarks": [[float(i) * 0.01, 0.5, 0.2] for i in range(21)],
        }
        ws.send_json(valid_frame)
        msg1 = ws.receive_json()
        assert msg1["label"] == "A"
        assert msg1["state"] == "DETECTING"

        # 2. Send invalid frame (e.g. wrong points) -> returns error and keeps connection open
        invalid_frame = {
            "session_id": "ws-test-session",
            "language": "ASL",
            "handedness": "Right",
            "landmarks": [[0.1, 0.2, 0.3]] * 5,
        }
        ws.send_json(invalid_frame)
        msg_err = ws.receive_json()
        assert "error" in msg_err

        # 3. Connection is still alive, send valid frame again
        ws.send_json(valid_frame)
        msg3 = ws.receive_json()
        assert msg3["label"] == "A"
