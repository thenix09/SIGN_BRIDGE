"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health endpoint response payload."""

    model_config = {"protected_namespaces": ()}

    status: str = Field(default="ok", description="Service status")
    model_loaded: bool = Field(
        description="Whether the letter recognition model file exists/loaded"
    )
    word_model_loaded: bool = Field(
        description="Whether the word recognition model file exists/loaded"
    )
    version: str = Field(default="1.0.0", description="Application version")


class ErrorResponse(BaseModel):
    """Standardized error response payload."""

    error: str = Field(description="Machine-readable error code")
    detail: str = Field(description="Human-readable error description")
