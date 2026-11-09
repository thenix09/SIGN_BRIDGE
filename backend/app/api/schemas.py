"""Pydantic schemas for API request and response validation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HealthResponse(BaseModel):
    """Health endpoint response payload."""

    model_config = ConfigDict(protected_namespaces=())

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


class RecognizeRequest(BaseModel):
    """Payload for fingerspelling recognition."""

    session_id: str = Field(description="Session identifier")
    language: str = Field(default="ASL", description="Sign language code (e.g. 'ASL')")
    handedness: Literal["Left", "Right"] = Field(default="Right", description="Detected hand")
    landmarks: list[list[float]] = Field(description="21 landmarks with [x, y, z] coordinates each")

    @field_validator("landmarks")
    @classmethod
    def validate_landmarks(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 21:
            raise ValueError(f"Fingerspelling requires exactly 21 landmarks, got {len(v)}")
        for i, pt in enumerate(v):
            if len(pt) < 3:
                raise ValueError(f"Point {i} requires at least 3 coordinates [x, y, z]")
        return v


class RecognizeResponse(BaseModel):
    """Recognition result returned per video frame."""

    label: str = Field(description="Recognized candidate letter or empty")
    confidence: float = Field(description="Confidence probability [0.0, 1.0]")
    state: str = Field(description="Stability state: IDLE, DETECTING, STABILISING, CONFIRMED")
    confirmed_letter: str | None = Field(default=None, description="Newly confirmed letter")
    current_word: str = Field(description="Word currently being spelled")
    sentence: str = Field(description="Full accumulated sentence")
    suggestions: list[str] = Field(
        default_factory=list, description="Top autocomplete word suggestions"
    )


class SessionResetRequest(BaseModel):
    """Payload to reset a session."""

    session_id: str


class SessionActionRequest(BaseModel):
    """Payload for session backspace or commit."""

    session_id: str
    word: str | None = None


class ModeStatus(BaseModel):
    """Status for a single recognition mode."""

    mode: Literal["spell", "words"]
    status: Literal["available", "experimental", "withheld", "unavailable"]


class LanguageItem(BaseModel):
    """Status entry for a sign language."""

    code: str
    modes: list[ModeStatus]


class LanguagesResponse(BaseModel):
    """Supported sign languages and mode availability."""

    languages: list[LanguageItem]


class OkResponse(BaseModel):
    """Generic success response."""

    ok: bool = True
