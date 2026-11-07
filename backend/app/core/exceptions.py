"""Domain and application exceptions."""


class SignBridgeError(Exception):
    """Base exception class for all SignBridge errors."""

    def __init__(
        self, message: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR"
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class InvalidLandmarkError(SignBridgeError):
    """Raised when landmark data fails shape, length, coordinate or finite value validation."""

    def __init__(self, message: str = "Invalid landmark input") -> None:
        super().__init__(message=message, status_code=400, error_code="INVALID_LANDMARK")


class UnsupportedLanguageError(SignBridgeError):
    """Raised when an unsupported sign language is requested."""

    def __init__(self, message: str = "Requested sign language is not supported") -> None:
        super().__init__(message=message, status_code=400, error_code="UNSUPPORTED_LANGUAGE")


class ModelNotAvailableError(SignBridgeError):
    """Raised when a model or resource required for recognition is not loaded or available."""

    def __init__(self, message: str = "Model is not loaded or available") -> None:
        super().__init__(message=message, status_code=503, error_code="MODEL_NOT_AVAILABLE")


class SessionNotFoundError(SignBridgeError):
    """Raised when a requested session ID does not exist or has expired."""

    def __init__(self, message: str = "Session not found or expired") -> None:
        super().__init__(message=message, status_code=404, error_code="SESSION_NOT_FOUND")
