"""Text-to-sign conversion service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.app.core.events import EventBus, default_event_bus
from backend.app.rendering.base import SignFrame
from backend.app.rendering.factory import RendererFactory
from backend.app.rendering.renderers import FallbackSignRenderer


@dataclass(frozen=True)
class ConversionResult:
    """Result of converting text into sign animation playback data."""

    frames: list[SignFrame]
    segments: list[dict[str, Any]]
    fallback_tokens: list[str]


class TextToSignService:
    """Orchestrates converting text sentences into animated sign frames."""

    def __init__(
        self,
        renderer_factory: type[RendererFactory] | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.renderer_factory = renderer_factory or RendererFactory
        self.bus = bus or default_event_bus

    def convert(self, language: str, text: str) -> ConversionResult:
        """Convert input text into playback sign frames and segments.

        Args:
            language: Target sign language code (e.g. 'ASL').
            text: Text string to convert (max 200 characters).

        Returns:
            ConversionResult containing frames, token segments, and fingerspelled fallbacks.

        Raises:
            ValueError: If text exceeds 200 characters.
        """
        if len(text) > 200:
            raise ValueError(f"Text length ({len(text)}) exceeds maximum allowed limit of 200 characters")

        renderer = self.renderer_factory.create("auto")
        if isinstance(renderer, FallbackSignRenderer):
            frames, segments, fallback_tokens = renderer.render_with_metadata(text)
        else:
            frames = renderer.render(text)
            segments = []
            fallback_tokens = []

        if frames:
            self.bus.publish(
                "text_converted",
                {
                    "text": text,
                    "language": language,
                },
            )

        return ConversionResult(
            frames=frames,
            segments=segments,
            fallback_tokens=fallback_tokens,
        )
