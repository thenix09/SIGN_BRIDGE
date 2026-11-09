"""Data stores for letter templates and word playback lexicons."""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.rendering.base import SignFrame


class TemplateStore:
    """Provides access to 21x3 mean landmark templates for fingerspelling letters."""

    def __init__(self, templates_path: str | Path | None = None) -> None:
        settings = get_settings()
        self.path = Path(templates_path or settings.templates_path)
        self._templates: dict[str, list[list[float]]] = {}
        self._load()

    def _load(self) -> None:
        """Load templates from disk if available."""
        if not self.path.is_file():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._templates = {
                    k.upper(): v for k, v in data.items() if isinstance(v, list)
                }
        except Exception:
            self._templates = {}

    def has(self, letter: str) -> bool:
        """Return True if template exists for the given letter."""
        return letter.upper() in self._templates

    def get(self, letter: str) -> list[list[float]]:
        """Return 21x3 mean landmark coordinates for the given letter.

        Raises:
            KeyError: If letter template is not found.
        """
        clean = letter.upper()
        if clean not in self._templates:
            raise KeyError(f"No template available for letter '{letter}'")
        return self._templates[clean]


class LexiconStore:
    """Provides access to recorded 48-frame medoid animation sequences for vocabulary words."""

    def __init__(self, lexicon_path: str | Path | None = None) -> None:
        settings = get_settings()
        self.path = Path(lexicon_path or settings.lexicon_path)
        self._lexicon: dict[str, list[SignFrame]] = {}
        self._load()

    def _load(self) -> None:
        """Load lexicon from disk if available."""
        if not self.path.is_file():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for word, raw_frames in data.items():
                    if isinstance(raw_frames, list):
                        frames = [
                            SignFrame(
                                label=str(rf.get("label", word)),
                                kind=rf.get("kind", "word"),
                                pose=rf.get("pose", []),
                                left_hand=rf.get("left_hand"),
                                right_hand=rf.get("right_hand"),
                                duration_ms=int(rf.get("duration_ms", 300)),
                            )
                            for rf in raw_frames
                            if isinstance(rf, dict)
                        ]
                        self._lexicon[word.lower()] = frames
        except Exception:
            self._lexicon = {}

    def has(self, word: str) -> bool:
        """Return True if playback sequence exists for the given word."""
        return word.lower() in self._lexicon

    def get(self, word: str) -> list[SignFrame]:
        """Return recorded animation frames for the given word.

        Raises:
            KeyError: If word is not present in lexicon.
        """
        clean = word.lower()
        if clean not in self._lexicon:
            raise KeyError(f"No lexicon entry available for word '{word}'")
        return self._lexicon[clean]
