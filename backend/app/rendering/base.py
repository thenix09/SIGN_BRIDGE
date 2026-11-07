"""Abstract base classes and fakes for the text-to-sign rendering subsystem."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SignFrame:
    """The canonical frame model for rendered sign animation."""

    label: str
    kind: Literal["word", "letter", "pause"]
    pose: list[list[float]]
    left_hand: list[list[float]] | None
    right_hand: list[list[float]] | None
    duration_ms: int


class SignRenderer(ABC):
    """Abstract strategy for converting text into an animation sequence of sign frames."""

    @abstractmethod
    def render(self, text: str) -> list[SignFrame]:
        """Render given text string into a list of SignFrame instances.

        Args:
            text: Text to convert into sign animation frames.

        Returns:
            List of SignFrame objects representing animation playback.
        """
        pass


class FakeSignRenderer(SignRenderer):
    """Fake sign renderer for testing and decoupled development."""

    def __init__(self, default_duration_ms: int = 300) -> None:
        self.default_duration_ms = default_duration_ms

    def render(self, text: str) -> list[SignFrame]:
        """Return dummy sign frames corresponding to characters in the text."""
        frames: list[SignFrame] = []
        for char in text.upper():
            if char.isalpha():
                frames.append(
                    SignFrame(
                        label=char,
                        kind="letter",
                        pose=[],
                        left_hand=None,
                        right_hand=[[0.0, 0.0] for _ in range(21)],
                        duration_ms=self.default_duration_ms,
                    )
                )
            elif char.isspace():
                frames.append(
                    SignFrame(
                        label="PAUSE",
                        kind="pause",
                        pose=[],
                        left_hand=None,
                        right_hand=None,
                        duration_ms=self.default_duration_ms,
                    )
                )
        return frames
