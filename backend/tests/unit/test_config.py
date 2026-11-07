"""Unit tests for configuration and settings."""

import os
from unittest.mock import patch

from backend.app.core.config import Settings, get_settings


def test_settings_defaults() -> None:
    """Verify default setting values match specifications."""
    settings = Settings()
    assert settings.confidence_threshold == 0.60
    assert settings.stability_required_frames == 8
    assert settings.stability_cooldown_frames == 10
    assert settings.session_ttl_seconds == 1800
    assert settings.word_gate == 0.75
    assert settings.max_request_bytes == 65536
    assert settings.database_url == "sqlite:///./signbridge.db"


def test_settings_env_override() -> None:
    """Verify environment variables with SIGNBRIDGE_ prefix override defaults."""
    with patch.dict(
        os.environ, {"SIGNBRIDGE_CONFIDENCE_THRESHOLD": "0.85", "SIGNBRIDGE_WORD_GATE": "0.70"}
    ):
        settings = Settings()
        assert settings.confidence_threshold == 0.85
        assert settings.word_gate == 0.70


def test_cors_origins_parsing() -> None:
    """Verify string parsing into list of origins."""
    settings = Settings(cors_origins="http://localhost:3000, https://signbridge.app")
    assert settings.cors_origins == ["http://localhost:3000", "https://signbridge.app"]


def test_get_settings_cached() -> None:
    """Verify get_settings returns cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
