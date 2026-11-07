"""Unit tests for custom exception classes."""

from backend.app.core.exceptions import (
    InvalidLandmarkError,
    ModelNotAvailableError,
    SessionNotFoundError,
    SignBridgeError,
    UnsupportedLanguageError,
)


def test_base_signbridge_error() -> None:
    """Verify base exception attributes."""
    err = SignBridgeError("Custom message", status_code=500, error_code="INTERNAL_ERROR")
    assert err.message == "Custom message"
    assert err.status_code == 500
    assert err.error_code == "INTERNAL_ERROR"
    assert str(err) == "Custom message"


def test_invalid_landmark_error() -> None:
    """Verify InvalidLandmarkError status and error code."""
    err = InvalidLandmarkError("Expected 21 points")
    assert err.status_code == 400
    assert err.error_code == "INVALID_LANDMARK"
    assert "Expected 21 points" in err.message


def test_unsupported_language_error() -> None:
    """Verify UnsupportedLanguageError status and error code."""
    err = UnsupportedLanguageError("Language BSL not supported")
    assert err.status_code == 400
    assert err.error_code == "UNSUPPORTED_LANGUAGE"


def test_model_not_available_error() -> None:
    """Verify ModelNotAvailableError status and error code."""
    err = ModelNotAvailableError("Model weights not found")
    assert err.status_code == 503
    assert err.error_code == "MODEL_NOT_AVAILABLE"


def test_session_not_found_error() -> None:
    """Verify SessionNotFoundError status and error code."""
    err = SessionNotFoundError("Session abc expired")
    assert err.status_code == 404
    assert err.error_code == "SESSION_NOT_FOUND"
