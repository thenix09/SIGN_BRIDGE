"""Domain model for training and evaluation samples."""

import math
from dataclasses import dataclass

from backend.app.core.exceptions import InvalidLandmarkError


@dataclass(frozen=True)
class Sample:
    """Represents a labeled dataset sample with 63 invariant features."""

    label: str
    group: str
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate feature count and finite values."""
        if len(self.features) != 63:
            raise InvalidLandmarkError(
                f"Sample features must have length 63, received {len(self.features)}"
            )
        for val in self.features:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise InvalidLandmarkError("Sample features must contain finite numbers")
