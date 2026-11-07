"""HandFrame domain model and landmark canonicalisation for Spell mode."""

import math
from collections.abc import Sequence

from backend.app.core.exceptions import InvalidLandmarkError
from backend.app.domain.landmark import Landmark


def canonicalise_landmarks(
    raw: Sequence[Sequence[float]],
    handedness: str,
    canonical: str,
) -> list[list[float]]:
    """Mirror raw landmarks if detected handedness does not match canonical.

    If handedness != canonical, x = 1.0 - x for every landmark point.
    Preserves y and z coordinates. Used by both dataset bootstrap and live API.

    Args:
        raw: Sequence of coordinate lists (each [x, y, z] or [x, y]).
        handedness: Handedness label returned by detector (e.g. 'Left', 'Right').
        canonical: Canonical handedness reference (e.g. 'Right').

    Returns:
        List of 3D points [x, y, z] canonicalized to match canonical handedness.
    """
    if not raw:
        return []

    should_mirror = handedness.strip().lower() != canonical.strip().lower()
    result: list[list[float]] = []

    for pt in raw:
        if len(pt) < 2:
            raise InvalidLandmarkError(f"Landmark point must contain at least x and y: {pt}")
        x = 1.0 - float(pt[0]) if should_mirror else float(pt[0])
        y = float(pt[1])
        z = float(pt[2]) if len(pt) > 2 else 0.0
        result.append([x, y, z])

    return result


class HandFrame:
    """Encapsulates 21 landmarks of a detected hand for fingerspelling recognition."""

    def __init__(self, landmarks: Sequence[Landmark]) -> None:
        """Construct a HandFrame from a sequence of exactly 21 Landmark objects.

        Args:
            landmarks: Sequence of exactly 21 Landmark instances.

        Raises:
            InvalidLandmarkError: If count != 21 or any item is invalid.
        """
        if len(landmarks) != 21:
            raise InvalidLandmarkError(
                f"HandFrame requires exactly 21 landmarks, received {len(landmarks)}"
            )
        for i, lm in enumerate(landmarks):
            if not isinstance(lm, Landmark):
                raise InvalidLandmarkError(f"Landmark at index {i} is not an instance of Landmark")

        self._landmarks: tuple[Landmark, ...] = tuple(landmarks)

    @property
    def landmarks(self) -> tuple[Landmark, ...]:
        """Return read-only tuple of 21 landmarks."""
        return self._landmarks

    @classmethod
    def from_raw(cls, raw: Sequence[Sequence[float]]) -> "HandFrame":
        """Construct a HandFrame from raw coordinate sequences.

        Args:
            raw: Sequence of 21 points, each having at least [x, y, z].

        Returns:
            Validated HandFrame instance.
        """
        if len(raw) != 21:
            raise InvalidLandmarkError(f"Raw input requires 21 points, received {len(raw)}")

        landmarks: list[Landmark] = []
        for i, pt in enumerate(raw):
            if len(pt) < 3:
                raise InvalidLandmarkError(
                    f"Point at index {i} must have at least 3 coordinates (x, y, z)"
                )
            landmarks.append(Landmark(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])))

        return cls(landmarks)

    def to_feature_vector(self) -> list[float]:
        """Convert hand landmarks to a 63-element invariant feature vector.

        Translates wrist (landmark index 0) to origin (0, 0, 0), normalizes by
        the maximum Euclidean distance from the wrist to any landmark, and
        flattens to 63 floats.

        Returns:
            List of 63 float features.
        """
        wrist = self._landmarks[0]

        # Compute wrist-relative coordinates
        rel_coords: list[tuple[float, float, float]] = [
            (lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in self._landmarks
        ]

        # Compute Euclidean distances from wrist
        distances = [math.sqrt(dx * dx + dy * dy + dz * dz) for dx, dy, dz in rel_coords]
        max_dist = max(distances)
        scale = max_dist if max_dist > 1e-7 else 1.0

        features: list[float] = []
        for dx, dy, dz in rel_coords:
            features.extend([dx / scale, dy / scale, dz / scale])

        return features
