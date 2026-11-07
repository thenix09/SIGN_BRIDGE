"""Integration test to verify all sample JSON fixtures exist and are valid JSON."""

import json
from pathlib import Path

FIXTURES_DIR = Path("backend/tests/fixtures/sample_responses")

REQUIRED_FIXTURES = [
    "health.json",
    "languages.json",
    "recognize.json",
    "recognize_words.json",
    "vocabulary.json",
    "session_reset.json",
    "session_backspace.json",
    "session_commit.json",
    "text_to_sign.json",
    "history_list.json",
    "history_delete.json",
    "samples_create.json",
    "samples_stats.json",
]


def test_all_sample_fixtures_exist_and_are_valid() -> None:
    """Verify that every endpoint fixture required by the frozen API contract exists."""
    assert FIXTURES_DIR.is_dir(), f"Fixtures directory {FIXTURES_DIR} does not exist"

    for filename in REQUIRED_FIXTURES:
        file_path = FIXTURES_DIR / filename
        assert file_path.exists(), f"Missing required sample fixture: {filename}"
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            assert data is not None
