"""Sign animation renderers for fingerspelling, word playback, and fallback composition."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.rendering.base import SignFrame, SignRenderer
from backend.app.rendering.stores import LexiconStore, TemplateStore

# Fixed chest anchor for letter frames in body space
CHEST_ANCHOR_X = 0.20
CHEST_ANCHOR_Y = 0.30
LETTER_DURATION_MS = 400
PAUSE_DURATION_MS = 200


class LandmarkTemplateRenderer(SignRenderer):
    """Renders text characters as static 2D landmark frames translated to a chest anchor."""

    def __init__(
        self,
        template_store: TemplateStore | None = None,
        duration_ms: int = LETTER_DURATION_MS,
        anchor_x: float = CHEST_ANCHOR_X,
        anchor_y: float = CHEST_ANCHOR_Y,
    ) -> None:
        self.template_store = template_store or TemplateStore()
        self.duration_ms = duration_ms
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y

    def render_char(self, char: str) -> SignFrame | None:
        """Render a single alphabetic character into a SignFrame anchored near the chest."""
        clean = char.upper()
        if not clean.isalpha() or not self.template_store.has(clean):
            return None

        raw_points = self.template_store.get(clean)
        translated_hand = [
            [round(pt[0] + self.anchor_x, 4), round(pt[1] + self.anchor_y, 4)]
            for pt in raw_points
        ]

        return SignFrame(
            label=clean,
            kind="letter",
            pose=[],
            left_hand=None,
            right_hand=translated_hand,
            duration_ms=self.duration_ms,
        )

    def render(self, text: str) -> list[SignFrame]:
        """Render all alphabetic characters in text, ignoring punctuation and spaces."""
        frames: list[SignFrame] = []
        for ch in text.upper():
            frame = self.render_char(ch)
            if frame is not None:
                frames.append(frame)
        return frames


class WordSequenceRenderer(SignRenderer):
    """Renders closed-vocabulary words as recorded 48-frame medoid animation sequences."""

    def __init__(self, lexicon_store: LexiconStore | None = None) -> None:
        self.lexicon_store = lexicon_store or LexiconStore()

    def render(self, text: str) -> list[SignFrame]:
        """Render word from lexicon if available."""
        clean = text.strip().lower()
        if self.lexicon_store.has(clean):
            return self.lexicon_store.get(clean)
        return []


class FallbackSignRenderer(SignRenderer):
    """Composite renderer: converts text into word sequences or fingerspelled fallbacks."""

    def __init__(
        self,
        word_renderer: WordSequenceRenderer | None = None,
        template_renderer: LandmarkTemplateRenderer | None = None,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.word_renderer = word_renderer or WordSequenceRenderer()
        self.template_renderer = template_renderer or LandmarkTemplateRenderer()
        self.aliases = aliases if aliases is not None else self._load_default_aliases()

    @staticmethod
    def _load_default_aliases() -> dict[str, str]:
        """Load aliases from vocabulary.json if available."""
        settings = get_settings()
        vocab_path = Path(settings.vocabulary_path)
        if vocab_path.is_file():
            try:
                with vocab_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "aliases" in data:
                    return {k.lower(): v.lower() for k, v in data["aliases"].items()}
            except Exception:
                pass
        return {"thank you": "thankyou", "thanks": "thankyou"}

    def _normalize_tokens(self, text: str) -> list[str]:
        """Clean punctuation, apply multi-word aliases, and extract ordered word tokens."""
        clean = text.lower().strip()
        # Drop punctuation except whitespace and alphanumerics
        clean = re.sub(r"[^\w\s]", "", clean)
        if not clean:
            return []

        # Replace multi-word aliases first (e.g. "thank you" -> "thankyou")
        for alias, canonical in self.aliases.items():
            if " " in alias and alias in clean:
                clean = re.sub(rf"\b{re.escape(alias)}\b", canonical, clean)

        raw_tokens = clean.split()
        tokens: list[str] = []
        for t in raw_tokens:
            mapped = self.aliases.get(t, t)
            tokens.append(mapped)
        return tokens

    def render_with_metadata(
        self, text: str
    ) -> tuple[list[SignFrame], list[dict[str, Any]], list[str]]:
        """Render text into animation frames along with token segments and fallback words.

        Args:
            text: Input sentence or words (max 200 characters).

        Returns:
            Tuple of:
            - frames: list of SignFrame instances (words, letters, pauses).
            - segments: list of {"token": str, "kind": "word"|"spelled", "start_index": int, "end_index": int}.
            - fallback_tokens: list of words that were fingerspelled due to missing lexicon.
        """
        tokens = self._normalize_tokens(text)
        if not tokens:
            return [], [], []

        frames: list[SignFrame] = []
        segments: list[dict[str, Any]] = []
        fallback_tokens: list[str] = []

        for idx, token in enumerate(tokens):
            start_idx = len(frames)

            # Check if token exists in lexicon
            if self.word_renderer.lexicon_store.has(token):
                word_frames = self.word_renderer.render(token)
                frames.extend(word_frames)
                end_idx = len(frames)
                segments.append(
                    {
                        "token": token,
                        "kind": "word",
                        "start_index": start_idx,
                        "end_index": end_idx,
                    }
                )
            else:
                # Fall back to fingerspelling letters
                fallback_tokens.append(token)
                letter_frames = self.template_renderer.render(token)
                frames.extend(letter_frames)
                end_idx = len(frames)
                segments.append(
                    {
                        "token": token,
                        "kind": "spelled",
                        "start_index": start_idx,
                        "end_index": end_idx,
                    }
                )

            # Insert pause frame between tokens (not after the final token)
            if idx < len(tokens) - 1:
                frames.append(
                    SignFrame(
                        label="PAUSE",
                        kind="pause",
                        pose=[],
                        left_hand=None,
                        right_hand=None,
                        duration_ms=PAUSE_DURATION_MS,
                    )
                )

        return frames, segments, fallback_tokens

    def render(self, text: str) -> list[SignFrame]:
        """Render animation frames for text."""
        frames, _, _ = self.render_with_metadata(text)
        return frames
