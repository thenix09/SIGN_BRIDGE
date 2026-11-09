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


class WordCandidate(BaseModel):
    """A single candidate prediction for word recognition."""

    label: str
    confidence: float


class WordFrameInput(BaseModel):
    """Input frame detections for sequence word recognition."""

    pose: list[list[float]] | None = None
    hands: list[list[list[float]]] = Field(default_factory=list)

    @field_validator("pose")
    @classmethod
    def validate_pose(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        if v is None:
            return None
        if len(v) not in (9, 17, 33):
            # Must have at least 9 or 17 points
            if len(v) < 9:
                raise ValueError(f"Pose requires at least 9 landmarks, got {len(v)}")
        for pt in v:
            if len(pt) < 2:
                raise ValueError("Pose points require at least [x, y]")
        return v

    @field_validator("hands")
    @classmethod
    def validate_hands(cls, v: list[list[list[float]]]) -> list[list[list[float]]]:
        if len(v) > 2:
            raise ValueError(f"At most 2 hands allowed, got {len(v)}")
        for h_idx, hand in enumerate(v):
            if len(hand) != 21:
                raise ValueError(f"Hand {h_idx} requires 21 landmarks, got {len(hand)}")
            for pt in hand:
                if len(pt) < 2:
                    raise ValueError("Hand points require at least [x, y]")
        return v


class RecognizeWordsRequest(BaseModel):
    """Request payload for word sequence recognition."""

    session_id: str
    language: str = "ASL"
    frame: WordFrameInput | None = None
    window: list[WordFrameInput] | None = None


class RecognizeWordsResponse(BaseModel):
    """Response returned for word sequence recognition."""

    state: str = Field(description="Segmenter/recognition state: WAITING, SIGNING, COMPLETE")
    top_k: list[WordCandidate] = Field(
        default_factory=list, description="Top-k gesture predictions"
    )
    accepted_word: str | None = Field(
        default=None, description="Accepted word if threshold & margin met"
    )
    sentence: str = Field(description="Accumulated sentence")
    message: str | None = Field(default=None, description="Error or rejection reason")


class VocabularyWord(BaseModel):
    """Closed vocabulary word entry."""

    label: str
    category: str


class VocabularyResponse(BaseModel):
    """Response for GET /api/vocabulary."""

    size: int
    status: Literal["available", "experimental", "withheld", "unavailable"]
    words: list[VocabularyWord]

