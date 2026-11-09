"""Application configuration and settings singleton."""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable override support.

    Prefix for environment overrides: SIGNBRIDGE_
    """

    model_config = SettingsConfigDict(
        env_prefix="SIGNBRIDGE_",
        env_file=".env",
        extra="ignore",
    )

    # Recognition thresholds
    confidence_threshold: float = Field(
        default=0.40,
        description="Confidence threshold for letter recognition",
    )
    stability_required_frames: int = Field(
        default=4,
        description="Required consecutive frames for letter confirmation",
    )
    stability_cooldown_frames: int = Field(
        default=6,
        description="Frames before the same letter can be confirmed again",
    )

    # File paths
    model_path: str = Field(
        default="backend/models/asl_model.joblib",
        description="Path to serialized ASL letter model",
    )
    templates_path: str = Field(
        default="backend/models/templates.json",
        description="Path to landmark templates",
    )
    letters_meta_path: str = Field(
        default="backend/models/letters_meta.json",
        description="Path to letter metadata (canonical handedness, etc.)",
    )
    word_model_path: str = Field(
        default="backend/models/word_model.onnx",
        description="Path to exported ONNX word recognition model",
    )
    word_vocab_path: str = Field(
        default="backend/models/word_vocab.json",
        description="Path to word model vocabulary",
    )
    lexicon_path: str = Field(
        default="backend/models/lexicon.json",
        description="Path to text-to-sign playback lexicon",
    )
    vocabulary_path: str = Field(
        default="backend/data/vocabulary.json",
        description="Path to closed vocabulary specification",
    )
    word_gate_path: str = Field(
        default="backend/models/word_gate.json",
        description="Path to word gate status file",
    )
    words_path: str = Field(
        default="backend/data/words.txt",
        description="Path to word dictionary for autocomplete suggestions",
    )

    # Database & Sessions
    database_url: str = Field(
        default="sqlite:///./signbridge.db",
        description="SQLAlchemy database connection URL",
    )
    session_ttl_seconds: int = Field(
        default=1800,
        description="Session idle expiration time in seconds (30 mins)",
    )

    # Word evaluation gate
    word_gate: float = Field(
        default=0.75,
        description="Accuracy gate threshold for word model (0.75 for 30 words, 0.70 for 50)",
    )

    # Security & Networking
    cors_origins: list[str] | str = Field(
        default=["*"],
        description="Permitted CORS origins or comma-separated list",
    )
    max_request_bytes: int = Field(
        default=65536,
        description="Maximum allowed HTTP request payload size in bytes",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        """Parse comma-separated strings into a list of origins if provided as a string."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
