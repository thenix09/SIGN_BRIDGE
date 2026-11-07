"""Domain model for a recognized gesture or sign."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Gesture:
    """Represents a recognized gesture candidate with a label and confidence score."""

    label: str
    confidence: float
