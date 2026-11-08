"""Sequence features and preprocessing for Words mode isolated sign recognition."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from backend.app.core.exceptions import InvalidLandmarkError

# Landmark indices and dimensions
POSE_INDICES = (0, 9, 10, 11, 12, 13, 14, 15, 16)  # nose, mouth L/R, shoulders, elbows, wrists
N_POSE = 9
N_HAND = 21
T = 32
N_FEATURES = 2 * (N_POSE + 2 * N_HAND) + 2  # 18 pose + 42 left + 42 right + 2 flags = 104

# Indices in the 9-pose subset
# 0: nose, 1: mouth L, 2: mouth R, 3: shoulder L, 4: shoulder R, 5: elbow L, 6: elbow R, 7: wrist L, 8: wrist R
POSE_SUBSET_PAIRS = [
    (1, 2),  # mouth L/R
    (3, 4),  # shoulder L/R
    (5, 6),  # elbow L/R
    (7, 8),  # wrist L/R
]


@dataclass(frozen=True)
class LandmarkFrame:
    """Represents a single frame of pose and hand coordinates for sequence processing."""

    pose: tuple[tuple[float, float], ...]  # exactly 9 (x, y) points
    left_hand: tuple[tuple[float, float], ...] | None  # 21 (x, y) points or None
    right_hand: tuple[tuple[float, float], ...] | None  # 21 (x, y) points or None

    def __post_init__(self) -> None:
        """Validate landmark counts and coordinates."""
        if len(self.pose) != N_POSE:
            raise InvalidLandmarkError(
                f"LandmarkFrame pose requires {N_POSE} points, got {len(self.pose)}"
            )
        if self.left_hand is not None and len(self.left_hand) != N_HAND:
            raise InvalidLandmarkError(
                f"LandmarkFrame left_hand requires {N_HAND} points, got {len(self.left_hand)}"
            )
        if self.right_hand is not None and len(self.right_hand) != N_HAND:
            raise InvalidLandmarkError(
                f"LandmarkFrame right_hand requires {N_HAND} points, got {len(self.right_hand)}"
            )

    @classmethod
    def from_detections(
        cls,
        pose: Sequence[Sequence[float]],
        hands: Sequence[Sequence[Sequence[float]]] | None = None,
    ) -> "LandmarkFrame":
        """Construct LandmarkFrame, extracting 9 pose points and slotting hands by nearest pose wrist.

        Args:
            pose: Full pose (>=17 points) or 9-point pose subset.
            hands: Unordered list of up to 2 hands (each 21 points of [x, y] or [x, y, z]).

        Returns:
            Validated LandmarkFrame with hands assigned to left or right wrist slots.
        """
        # Extract 9 pose points
        if len(pose) >= 17:
            pose_points = tuple((float(pose[idx][0]), float(pose[idx][1])) for idx in POSE_INDICES)
        elif len(pose) == N_POSE:
            pose_points = tuple((float(pt[0]), float(pt[1])) for pt in pose)
        else:
            raise InvalidLandmarkError(
                f"Pose must have at least 17 landmarks or exactly 9 points, got {len(pose)}"
            )

        left_wrist_pose = pose_points[7]
        right_wrist_pose = pose_points[8]

        left_hand: tuple[tuple[float, float], ...] | None = None
        right_hand: tuple[tuple[float, float], ...] | None = None

        if hands and len(hands) > 0:
            converted_hands: list[tuple[tuple[float, float], ...]] = []
            for h in hands[:2]:
                if len(h) != N_HAND:
                    raise InvalidLandmarkError(
                        f"Detected hand must have 21 landmarks, got {len(h)}"
                    )
                converted_hands.append(tuple((float(pt[0]), float(pt[1])) for pt in h))

            if len(converted_hands) == 1:
                h_wrist = converted_hands[0][0]
                d_left = math.hypot(
                    h_wrist[0] - left_wrist_pose[0], h_wrist[1] - left_wrist_pose[1]
                )
                d_right = math.hypot(
                    h_wrist[0] - right_wrist_pose[0], h_wrist[1] - right_wrist_pose[1]
                )
                if d_left <= d_right:
                    left_hand = converted_hands[0]
                else:
                    right_hand = converted_hands[0]
            elif len(converted_hands) >= 2:
                h0_wrist = converted_hands[0][0]
                h1_wrist = converted_hands[1][0]

                cost1 = math.hypot(
                    h0_wrist[0] - left_wrist_pose[0], h0_wrist[1] - left_wrist_pose[1]
                ) + math.hypot(h1_wrist[0] - right_wrist_pose[0], h1_wrist[1] - right_wrist_pose[1])
                cost2 = math.hypot(
                    h0_wrist[0] - right_wrist_pose[0], h0_wrist[1] - right_wrist_pose[1]
                ) + math.hypot(h1_wrist[0] - left_wrist_pose[0], h1_wrist[1] - left_wrist_pose[1])

                if cost1 <= cost2:
                    left_hand = converted_hands[0]
                    right_hand = converted_hands[1]
                else:
                    left_hand = converted_hands[1]
                    right_hand = converted_hands[0]

        return cls(pose=pose_points, left_hand=left_hand, right_hand=right_hand)


def resample_sequence(seq: np.ndarray, length: int = T) -> np.ndarray:
    """Linearly interpolate sequence of shape (N, F) to exactly (length, F).

    Args:
        seq: Array of shape (N, F).
        length: Target temporal length (default T = 32).

    Returns:
        Interpolated array of shape (length, F).
    """
    n_frames, n_features = seq.shape
    if n_frames == length:
        return seq.copy()

    if n_frames == 1:
        return np.repeat(seq, length, axis=0)

    orig_indices = np.linspace(0, n_frames - 1, num=n_frames)
    target_indices = np.linspace(0, n_frames - 1, num=length)

    resampled = np.empty((length, n_features), dtype=seq.dtype)
    for feat_idx in range(n_features):
        resampled[:, feat_idx] = np.interp(target_indices, orig_indices, seq[:, feat_idx])

    return resampled


def canonicalise_sequence(seq: np.ndarray) -> np.ndarray:
    """Apply dominant-hand canonicalisation to sequence of shape (T, 104).

    Computes motion energy of left wrist vs right wrist. If the left wrist
    has greater energy, mirrors the sequence (x -> -x, swap left and right hands,
    swap symmetric pose points) so that the dominant hand is consistently right.
    A sequence and its mirror produce the exact same tensor.

    Args:
        seq: Array of shape (T, 104).

    Returns:
        Canonicalized array of shape (T, 104).
    """
    out = seq.copy()

    # Left hand wrist is feature 18, 19. Presence flag is feature 102.
    # Right hand wrist is feature 60, 61. Presence flag is feature 103.
    lw_x = out[:, 18]
    lw_y = out[:, 19]
    lw_present = out[:, 102] > 0.5

    rw_x = out[:, 60]
    rw_y = out[:, 61]
    rw_present = out[:, 103] > 0.5

    energy_left = 0.0
    energy_right = 0.0

    for t in range(1, len(out)):
        if lw_present[t] and lw_present[t - 1]:
            energy_left += math.hypot(lw_x[t] - lw_x[t - 1], lw_y[t] - lw_y[t - 1])
        if rw_present[t] and rw_present[t - 1]:
            energy_right += math.hypot(rw_x[t] - rw_x[t - 1], rw_y[t] - rw_y[t - 1])

    if energy_left > energy_right:
        # 1. Flip x coordinates for pose and both hands
        # Pose x coordinates: indices 0, 2, 4, 6, 8, 10, 12, 14, 16
        for p_idx in range(N_POSE):
            out[:, 2 * p_idx] = -out[:, 2 * p_idx]

        # 2. Swap symmetric pose pairs
        for p1, p2 in POSE_SUBSET_PAIRS:
            tmp_x = out[:, 2 * p1].copy()
            tmp_y = out[:, 2 * p1 + 1].copy()
            out[:, 2 * p1] = out[:, 2 * p2]
            out[:, 2 * p1 + 1] = out[:, 2 * p2 + 1]
            out[:, 2 * p2] = tmp_x
            out[:, 2 * p2 + 1] = tmp_y

        # 3. Flip x and swap left hand (18:60) and right hand (60:102)
        lh_slice = out[:, 18:60].copy()
        rh_slice = out[:, 60:102].copy()

        # Invert x
        lh_slice[:, 0::2] = -lh_slice[:, 0::2]
        rh_slice[:, 0::2] = -rh_slice[:, 0::2]

        out[:, 18:60] = rh_slice
        out[:, 60:102] = lh_slice

        # 4. Swap presence flags
        tmp_flag = out[:, 102].copy()
        out[:, 102] = out[:, 103]
        out[:, 103] = tmp_flag

    return out


class LandmarkWindow:
    """Window of sequential LandmarkFrames with transformation to normalized feature tensor."""

    def __init__(self, frames: Sequence[LandmarkFrame]) -> None:
        """Construct LandmarkWindow from a sequence of LandmarkFrames.

        Args:
            frames: Sequence of LandmarkFrame instances.

        Raises:
            InvalidLandmarkError: If frames sequence is empty or items invalid.
        """
        if not frames:
            raise InvalidLandmarkError("LandmarkWindow requires at least one LandmarkFrame")
        for idx, f in enumerate(frames):
            if not isinstance(f, LandmarkFrame):
                raise InvalidLandmarkError(f"Frame at index {idx} is not a LandmarkFrame")

        self._frames = list(frames)

    @property
    def frames(self) -> tuple[LandmarkFrame, ...]:
        """Return frames in the window."""
        return tuple(self._frames)

    def to_tensor(self) -> np.ndarray:
        """Transform frames into normalized, canonicalized, resampled tensor of shape (32, 104).

        Steps:
            1. Trim leading and trailing frames with no detected hands.
            2. Normalize coordinates: shoulder midpoint origin, shoulder distance scale.
            3. Resample sequence linearly to T = 32 frames.
            4. Canonicalize dominant hand (mirror left dominant to right).

        Returns:
            np.ndarray of shape (32, 104) with dtype float32.
        """
        frames = self._frames

        # 1. Trim leading and trailing frames without hands
        has_hands = [f.left_hand is not None or f.right_hand is not None for f in frames]

        if any(has_hands):
            first_idx = has_hands.index(True)
            last_idx = len(has_hands) - 1 - has_hands[::-1].index(True)
            trimmed_frames = frames[first_idx : last_idx + 1]
        else:
            trimmed_frames = frames

        # 2. Extract and normalize per-frame feature vectors
        raw_rows: list[list[float]] = []

        for f in trimmed_frames:
            # Midpoint of shoulders: subset index 3 (left shoulder) and 4 (right shoulder)
            sl_x, sl_y = f.pose[3]
            sr_x, sr_y = f.pose[4]

            ox = (sl_x + sr_x) / 2.0
            oy = (sl_y + sr_y) / 2.0
            scale = math.hypot(sl_x - sr_x, sl_y - sr_y)
            if scale < 1e-7:
                scale = 1.0

            row: list[float] = []

            # Pose: 9 points (18 floats)
            for px, py in f.pose:
                row.extend([(px - ox) / scale, (py - oy) / scale])

            # Left hand: 21 points (42 floats)
            if f.left_hand is not None:
                for lx, ly in f.left_hand:
                    row.extend([(lx - ox) / scale, (ly - oy) / scale])
            else:
                row.extend([0.0] * (2 * N_HAND))

            # Right hand: 21 points (42 floats)
            if f.right_hand is not None:
                for rx, ry in f.right_hand:
                    row.extend([(rx - ox) / scale, (ry - oy) / scale])
            else:
                row.extend([0.0] * (2 * N_HAND))

            # Presence flags (2 floats)
            row.append(1.0 if f.left_hand is not None else 0.0)
            row.append(1.0 if f.right_hand is not None else 0.0)

            raw_rows.append(row)

        arr = np.array(raw_rows, dtype=np.float32)

        # 3. Resample to exactly T = 32 frames
        resampled = resample_sequence(arr, length=T)

        # 4. Canonicalise dominant hand
        canonical = canonicalise_sequence(resampled)

        return canonical.astype(np.float32)
