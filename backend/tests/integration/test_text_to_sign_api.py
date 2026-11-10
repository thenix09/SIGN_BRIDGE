"""Integration tests for POST /api/text-to-sign endpoint."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_text_to_sign_endpoint_valid_spelled() -> None:
    payload = {
        "language": "ASL",
        "text": "Hello world",
    }
    response = client.post("/api/text-to-sign", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "frames" in data
    assert "segments" in data
    assert "fallback_tokens" in data
    assert isinstance(data["frames"], list)
    assert len(data["frames"]) > 0

    first_frame = data["frames"][0]
    assert "label" in first_frame
    assert "kind" in first_frame
    assert first_frame["kind"] in ("word", "letter", "pause")
    assert "duration_ms" in first_frame


def test_text_to_sign_endpoint_max_length_exceeded() -> None:
    payload = {
        "language": "ASL",
        "text": "A" * 201,
    }
    response = client.post("/api/text-to-sign", json=payload)
    assert response.status_code == 422


def test_text_to_sign_endpoint_empty_text() -> None:
    payload = {
        "language": "ASL",
        "text": "",
    }
    response = client.post("/api/text-to-sign", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["frames"] == []
    assert data["segments"] == []
    assert data["fallback_tokens"] == []
