"""3D Landmark representation with strict coordinate validation."""

import math
from dataclasses import dataclass

from backend.app.core.exceptions import InvalidLandmarkError


@dataclass(frozen=True)
class Landmark:
    """Immutable 3D coordinate point."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        """Validate coordinates are finite numbers."""
        for val, name in ((self.x, "x"), (self.y, "y"), (self.z, "z")):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise InvalidLandmarkError(f"Landmark {name} must be a finite number, got {val}")
