"""Offline tests for Parquet preprocessing and tensor generation.

This module does not test model inference or end-to-end recognition.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.app.domain.sequence import (
    N_FEATURES,
    N_HAND,
    POSE_INDICES,
    T,
)
from backend.scripts.preprocess_words import parquet_to_tensor


def create_synthetic_parquet(
    path: Path,
    num_frames: int = 20,
    include_face: bool = True,
    include_all_pose: bool = True,
    left_hand_mode: str = "present",
    right_hand_mode: str = "present",
    dominant_motion: str = "right",
    ox: float = 0.5,
    oy: float = 0.5,
    shoulder_width: float = 0.2,
) -> Path:
    """Generate a valid synthetic Kaggle asl-signs Parquet file for offline testing.

    Args:
        path: Output destination path.
        num_frames: Number of temporal frames to generate.
        include_face: Whether to inject 468 face mesh landmarks per frame.
        include_all_pose: Whether to inject all 33 MediaPipe pose landmarks (0..32).
        left_hand_mode: 'present', 'absent', 'nan', or 'intermittent'.
        right_hand_mode: 'present', 'absent', 'nan', or 'intermittent'.
        dominant_motion: 'right' (right hand moves), 'left' (left hand moves), or 'none'.
        ox: Body center x coordinate.
        oy: Body center y coordinate.
        shoulder_width: Distance between left and right shoulders.

    Returns:
        The written Path.
    """
    records: list[dict[str, object]] = []

    sl_x = ox - shoulder_width / 2.0
    sr_x = ox + shoulder_width / 2.0
    s_y = oy

    pose_point_indices = list(range(33)) if include_all_pose else list(POSE_INDICES)

    for f in range(num_frames):
        t_norm = float(f) / max(num_frames - 1, 1)

        # Pose coordinates
        for p_idx in pose_point_indices:
            if p_idx == 11:  # left shoulder
                px, py = sl_x, s_y
            elif p_idx == 12:  # right shoulder
                px, py = sr_x, s_y
            elif p_idx == 0:  # nose
                px, py = ox, oy - 0.15
            elif p_idx == 9:  # mouth left
                px, py = ox - 0.03, oy - 0.08
            elif p_idx == 10:  # mouth right
                px, py = ox + 0.03, oy - 0.08
            elif p_idx == 13:  # left elbow
                px, py = sl_x - 0.05, s_y + 0.15
            elif p_idx == 14:  # right elbow
                px, py = sr_x + 0.05, s_y + 0.15
            elif p_idx == 15:  # left wrist
                px, py = sl_x - 0.08, s_y + 0.30
            elif p_idx == 16:  # right wrist
                px, py = sr_x + 0.08, s_y + 0.30
            else:
                # Other non-essential pose points
                px, py = ox + float(p_idx) * 0.005, oy + 0.40

            records.append({
                "frame": f,
                "row_id": f"{f}-pose-{p_idx}",
                "type": "pose",
                "landmark_index": p_idx,
                "x": float(px),
                "y": float(py),
                "z": 0.0,
            })

        # Face landmarks (468 points) — privacy discard test
        if include_face:
            for face_idx in range(468):
                records.append({
                    "frame": f,
                    "row_id": f"{f}-face-{face_idx}",
                    "type": "face",
                    "landmark_index": face_idx,
                    "x": float(ox + (face_idx % 20) * 0.002),
                    "y": float(oy - 0.1 + (face_idx // 20) * 0.002),
                    "z": 0.0,
                })

        # Left hand
        if left_hand_mode != "absent":
            has_left = True
            if left_hand_mode == "intermittent" and (f % 2 == 0):
                has_left = False

            if has_left:
                base_lx = sl_x - 0.08
                base_ly = s_y + 0.30
                if dominant_motion == "left":
                    base_lx += 0.20 * math.sin(t_norm * math.pi)
                    base_ly += 0.20 * math.cos(t_norm * math.pi)

                for h_idx in range(N_HAND):
                    if left_hand_mode == "nan" and h_idx == 0:
                        lx_val, ly_val = np.nan, np.nan
                    else:
                        lx_val = base_lx + float(h_idx) * 0.003
                        ly_val = base_ly + float(h_idx) * 0.003

                    records.append({
                        "frame": f,
                        "row_id": f"{f}-left_hand-{h_idx}",
                        "type": "left_hand",
                        "landmark_index": h_idx,
                        "x": float(lx_val),
                        "y": float(ly_val),
                        "z": 0.0,
                    })

        # Right hand
        if right_hand_mode != "absent":
            has_right = True
            if right_hand_mode == "intermittent" and (f % 2 != 0):
                has_right = False

            if has_right:
                base_rx = sr_x + 0.08
                base_ry = s_y + 0.30
                if dominant_motion == "right":
                    base_rx += 0.20 * math.sin(t_norm * math.pi)
                    base_ry += 0.20 * math.cos(t_norm * math.pi)

                for h_idx in range(N_HAND):
                    if right_hand_mode == "nan" and h_idx == 0:
                        rx_val, ry_val = np.nan, np.nan
                    else:
                        rx_val = base_rx + float(h_idx) * 0.003
                        ry_val = base_ry + float(h_idx) * 0.003

                    records.append({
                        "frame": f,
                        "row_id": f"{f}-right_hand-{h_idx}",
                        "type": "right_hand",
                        "landmark_index": h_idx,
                        "x": float(rx_val),
                        "y": float(ry_val),
                        "z": 0.0,
                    })

    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(path))
    return path


def test_parquet_to_tensor_standard(tmp_path: Path) -> None:
    """Verify standard extraction produces exactly (32, 104) float32 finite tensor."""
    pq_path = tmp_path / "sample.parquet"
    create_synthetic_parquet(pq_path, num_frames=25)

    tensor = parquet_to_tensor(pq_path)

    assert tensor.shape == (T, N_FEATURES)
    assert tensor.shape == (32, 104)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_face_landmarks_are_discarded(tmp_path: Path) -> None:
    """Verify 468 face mesh landmarks are ignored and do not affect feature dimensions."""
    pq_with_face = tmp_path / "with_face.parquet"
    pq_without_face = tmp_path / "no_face.parquet"

    create_synthetic_parquet(pq_with_face, num_frames=15, include_face=True)
    create_synthetic_parquet(pq_without_face, num_frames=15, include_face=False)

    tensor_with = parquet_to_tensor(pq_with_face)
    tensor_without = parquet_to_tensor(pq_without_face)

    assert tensor_with.shape == (32, 104)
    assert tensor_without.shape == (32, 104)
    # Output must be identical since face landmarks are discarded before processing
    assert np.allclose(tensor_with, tensor_without, atol=1e-6)


def test_only_approved_nine_pose_landmarks_used(tmp_path: Path) -> None:
    """Verify that pose landmarks outside the 9 approved indices do not affect output."""
    pq_33_pose = tmp_path / "all_pose.parquet"
    pq_9_pose = tmp_path / "nine_pose.parquet"

    create_synthetic_parquet(pq_33_pose, num_frames=16, include_all_pose=True)
    create_synthetic_parquet(pq_9_pose, num_frames=16, include_all_pose=False)

    tensor_all = parquet_to_tensor(pq_33_pose)
    tensor_nine = parquet_to_tensor(pq_9_pose)

    assert tensor_all.shape == (32, 104)
    assert tensor_nine.shape == (32, 104)
    assert np.allclose(tensor_all, tensor_nine, atol=1e-6)


def test_missing_hands_preserved_and_flagged(tmp_path: Path) -> None:
    """Verify missing/absent hands are zero-filled and presence flags are set to 0.0."""
    pq_no_left = tmp_path / "no_left.parquet"
    create_synthetic_parquet(
        pq_no_left,
        num_frames=20,
        left_hand_mode="absent",
        right_hand_mode="present",
        dominant_motion="right",
    )

    tensor = parquet_to_tensor(pq_no_left)

    # In canonicalized tensor, absent left hand has presence flag 0.0 at index 102
    # Left hand slice is features 18:60
    assert np.all(tensor[:, 18:60] == 0.0)
    assert np.all(tensor[:, 102] == 0.0)
    # Right hand is present: presence flag is 1.0 at index 103
    assert np.all(tensor[:, 103] == 1.0)


def test_nan_hand_coordinates_treated_as_absent(tmp_path: Path) -> None:
    """Verify that NaN hand coordinates are safely treated as absent rather than crashing."""
    pq_nan_hand = tmp_path / "nan_hand.parquet"
    create_synthetic_parquet(
        pq_nan_hand,
        num_frames=20,
        left_hand_mode="nan",
        right_hand_mode="present",
        dominant_motion="right",
    )

    tensor = parquet_to_tensor(pq_nan_hand)

    assert tensor.shape == (32, 104)
    assert np.isfinite(tensor).all()
    # Left hand had NaN, so it must be treated as absent
    assert np.all(tensor[:, 18:60] == 0.0)
    assert np.all(tensor[:, 102] == 0.0)


def test_nan_pose_raises_error(tmp_path: Path) -> None:
    """Verify NaN pose coordinates raise ValueError because body reference cannot be established."""
    pq_path = tmp_path / "nan_pose.parquet"
    records = []
    for f in range(5):
        for p in POSE_INDICES:
            x_val = np.nan if p == 11 and f == 0 else 0.5
            records.append({
                "frame": f,
                "row_id": f"{f}-pose-{p}",
                "type": "pose",
                "landmark_index": p,
                "x": x_val,
                "y": 0.5,
                "z": 0.0,
            })
    df = pd.DataFrame(records)
    pq.write_table(pa.Table.from_pandas(df), str(pq_path))

    with pytest.raises(ValueError, match="NaN pose coordinates"):
        parquet_to_tensor(pq_path)


@pytest.mark.parametrize("num_frames", [4, 15, 32, 60])
def test_variable_sequence_lengths(tmp_path: Path, num_frames: int) -> None:
    """Verify variable sequence lengths resample to exactly (32, 104)."""
    pq_path = tmp_path / f"len_{num_frames}.parquet"
    create_synthetic_parquet(pq_path, num_frames=num_frames)

    tensor = parquet_to_tensor(pq_path)

    assert tensor.shape == (32, 104)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


def test_dominant_hand_mirror_canonicalization(tmp_path: Path) -> None:
    """Verify left-dominant sequences are mirrored to match right-dominant equivalents."""
    pq_right = tmp_path / "right_dominant.parquet"
    pq_left = tmp_path / "left_dominant.parquet"

    # Create right-dominant motion
    create_synthetic_parquet(
        pq_right,
        num_frames=20,
        left_hand_mode="present",
        right_hand_mode="present",
        dominant_motion="right",
    )
    # Create left-dominant motion
    create_synthetic_parquet(
        pq_left,
        num_frames=20,
        left_hand_mode="present",
        right_hand_mode="present",
        dominant_motion="left",
    )

    t_right = parquet_to_tensor(pq_right)
    t_left = parquet_to_tensor(pq_left)

    # In both, dominant motion should occupy the canonical right-hand feature slice.
    assert np.all(t_right[:, 103] == 1.0)
    assert np.all(t_left[:, 103] == 1.0)
    assert np.all(t_right[:, 102] == 1.0)
    assert np.all(t_left[:, 102] == 1.0)

    # Right-dominant motion moves toward increasing x; mirrored left-dominant
    # motion moves toward decreasing x after canonicalization.
    right_x = t_right[:, 60]
    left_x = t_left[:, 60]
    assert np.max(right_x) - right_x[0] > 0.1
    assert left_x[0] - np.min(left_x) > 0.1

    # The non-dominant left-hand landmark remains stationary in both tensors.
    assert np.ptp(t_right[:, 18]) < 1e-6
    assert np.ptp(t_left[:, 18]) < 1e-6


def test_missing_or_empty_parquet(tmp_path: Path) -> None:
    """Verify appropriate exceptions for missing or empty parquet files."""
    with pytest.raises(FileNotFoundError):
        parquet_to_tensor(tmp_path / "nonexistent.parquet")

    empty_path = tmp_path / "empty.parquet"
    empty_path.touch()
    with pytest.raises(FileNotFoundError):
        parquet_to_tensor(empty_path)
