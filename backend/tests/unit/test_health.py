"""Unit and API route tests for /api/health and middleware."""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.api.deps import get_app_settings
from backend.app.core.config import Settings
from backend.app.core.exceptions import InvalidLandmarkError
from backend.app.main import create_app


def test_health_endpoint_no_models(tmp_path: Path) -> None:
    """Verify /api/health returns ok and model_loaded false when models are missing."""
    custom_settings = Settings(
        model_path=str(tmp_path / "missing_asl_model.joblib"),
        word_model_path=str(tmp_path / "missing_word_model.onnx"),
    )
    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: custom_settings
    client = TestClient(app)

    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False
    assert data["word_model_loaded"] is False
    assert data["version"] == "1.0.0"


def test_health_endpoint_with_models(tmp_path: Path) -> None:
    """Verify /api/health reflects loaded models when files exist."""
    fake_model = tmp_path / "asl_model.joblib"
    fake_word_model = tmp_path / "word_model.onnx"
    fake_model.touch()
    fake_word_model.touch()

    custom_settings = Settings(
        model_path=str(fake_model),
        word_model_path=str(fake_word_model),
    )

    app = create_app()
    app.dependency_overrides[get_app_settings] = lambda: custom_settings

    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["word_model_loaded"] is True


def test_payload_size_limit_middleware() -> None:
    """Verify middleware rejects payloads larger than max_request_bytes with 413."""
    app = create_app()
    client = TestClient(app)

    oversized_payload = "x" * 70000
    response = client.post(
        "/api/health",
        content=oversized_payload,
        headers={"Content-Length": str(len(oversized_payload))},
    )
    assert response.status_code == 413
    assert response.json()["error"] == "PAYLOAD_TOO_LARGE"


def test_custom_exception_handler() -> None:
    """Verify custom SignBridgeError produces standardized JSON response."""
    from fastapi.routing import APIRoute

    async def trigger_error() -> None:
        raise InvalidLandmarkError("Test invalid landmark details")

    app = create_app()
    app.router.routes.insert(0, APIRoute("/test-error", trigger_error, methods=["GET"]))

    client = TestClient(app)
    response = client.get("/test-error")
    assert response.status_code == 400
    assert response.json() == {
        "error": "INVALID_LANDMARK",
        "detail": "Test invalid landmark details",
    }
