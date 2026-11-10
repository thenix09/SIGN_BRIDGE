"""Factory for instantiating SignRenderer strategies."""

from __future__ import annotations

from backend.app.rendering.base import SignRenderer
from backend.app.rendering.renderers import (
    FallbackSignRenderer,
    LandmarkTemplateRenderer,
    WordSequenceRenderer,
)


class RendererFactory:
    """Factory creating configured SignRenderer instances."""

    @classmethod
    def create(cls, kind: str = "auto") -> SignRenderer:
        """Create and return a SignRenderer strategy.

        Args:
            kind: "auto" (default, fallback composition), "letters", or "words".

        Returns:
            SignRenderer instance.
        """
        clean_kind = kind.lower().strip()
        if clean_kind == "letters":
            return LandmarkTemplateRenderer()
        elif clean_kind == "words":
            return WordSequenceRenderer()
        return FallbackSignRenderer()
