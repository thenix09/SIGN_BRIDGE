"""Unit tests for Words mode sequence features and preprocessing in domain/sequence.py."""

import math

import numpy as np
import pytest

from backend.app.core.exceptions import InvalidLandmarkError
from backend.app.domain.sequence import (
    N_FEATURES,
    N_POSE,
    LandmarkFrame,
    LandmarkWindow,
    T,
)


def _create_mock_pose(
    ox: float = 0.5, oy: float = 0.5, scale: float = 0.2
) -> tuple[tuple[float, float], ...]:
    """Create 9 mock pose points centered at (ox, oy)."""
    # 0: nose, 1: mouth L, 2: mouth R, 3: sh L, 4: sh R, 5: elb L, 6: elb R, 7: wr L, 8: wr R
    return (
        (ox, oy - scale),  # nose
        (ox - 0.2 * scale, oy - 0.7 * scale),  # mouth L
        (ox + 0.2 * scale, oy - 0.7 * scale),  # mouth R
        (ox - scale, oy),  # shoulder L
        (ox + scale, oy),  # shoulder R
        (ox - 1.2 * scale, oy + scale),  # elbow L
        (ox + 1.2 * scale, oy + scale),  # elbow R
        (ox - 1.4 * scale, oy + 1.5 * scale),  # wrist L
        (ox + 1.4 * scale, oy + 1.5 * scale),  # wrist R
    )


def _create_mock_hand(
    wrist_x: float, wrist_y: float, scale: float = 1.0
) -> tuple[tuple[float, float], ...]:
    """Create 21 mock hand points starting at (wrist_x, wrist_y)."""
    return tuple(
        (wrist_x + float(i) * 0.005 * scale, wrist_y + float(i) * 0.005 * scale) for i in range(21)
    )


def test_landmark_frame_validation() -> None:
    """Verify LandmarkFrame validates pose and hand lengths."""
    pose = _create_mock_pose()
    hand = _create_mock_hand(0.2, 0.8)

    # Valid frame
    frame = LandmarkFrame(pose=pose, left_hand=hand, right_hand=None)
    assert len(frame.pose) == 9
    assert frame.left_hand is not None
    assert frame.right_hand is None

    # Invalid pose size
    with pytest.raises(InvalidLandmarkError):
        LandmarkFrame(pose=pose[:8], left_hand=hand, right_hand=None)

    # Invalid hand size
    with pytest.raises(InvalidLandmarkError):
        LandmarkFrame(pose=pose, left_hand=hand[:20], right_hand=None)


def test_slot_assignment_by_nearest_wrist() -> None:
    """Verify from_detections assigns hands to nearest pose wrist slot."""
    pose = _create_mock_pose(ox=0.5, oy=0.5, scale=0.2)
    # Pose left wrist is at (0.5 - 1.4*0.2, 0.5 + 1.5*0.2) = (0.22, 0.8)
    # Pose right wrist is at (0.5 + 1.4*0.2, 0.5 + 1.5*0.2) = (0.78, 0.8)

    hand_near_left = [[0.23, 0.81] for _ in range(21)]
    hand_near_right = [[0.77, 0.79] for _ in range(21)]

    # Provide only hand near left
    f1 = LandmarkFrame.from_detections(pose, [hand_near_left])
    assert f1.left_hand is not None
    assert f1.right_hand is None

    # Provide only hand near right
    f2 = LandmarkFrame.from_detections(pose, [hand_near_right])
    assert f2.left_hand is None
    assert f2.right_hand is not None

    # Provide both hands in reverse order
    f3 = LandmarkFrame.from_detections(pose, [hand_near_right, hand_near_left])
    assert f3.left_hand is not None
    assert f3.right_hand is not None
    assert math.isclose(f3.left_hand[0][0], 0.23)
    assert math.isclose(f3.right_hand[0][0], 0.77)


def test_to_tensor_shape_and_dtype() -> None:
    """Verify LandmarkWindow.to_tensor produces shape (32, 104) and float32 dtype."""
    pose = _create_mock_pose()
    hand_l = _create_mock_hand(0.22, 0.8)
    hand_r = _create_mock_hand(0.78, 0.8)

    frames = [LandmarkFrame(pose=pose, left_hand=hand_l, right_hand=hand_r) for _ in range(20)]
    window = LandmarkWindow(frames)
    tensor = window.to_tensor()

    assert tensor.shape == (T, N_FEATURES)
    assert tensor.shape == (32, 104)
    assert tensor.dtype == np.float32


def test_trimming_and_resampling() -> None:
    """Verify leading and trailing frames with no hands are trimmed, then resampled to 32."""
    pose = _create_mock_pose()
    empty_frame = LandmarkFrame(pose=pose, left_hand=None, right_hand=None)
    hand_frame = LandmarkFrame(
        pose=pose,
        left_hand=None,
        right_hand=_create_mock_hand(0.78, 0.8),
    )

    # 5 empty, 10 with hand, 5 empty = 20 frames total
    raw_frames = [empty_frame] * 5 + [hand_frame] * 10 + [empty_frame] * 5
    window = LandmarkWindow(raw_frames)
    tensor = window.to_tensor()

    assert tensor.shape == (32, 104)
    # The right hand presence flag (feature 103) should be 1.0 across the resampled sequence
    assert np.all(tensor[:, 103] == 1.0)
    # Left hand flag (feature 102) should be 0.0
    assert np.all(tensor[:, 102] == 0.0)


def test_translation_and_scale_invariance() -> None:
    """Verify to_tensor is invariant to translation and camera scale."""
    pose1 = _create_mock_pose(ox=0.5, oy=0.5, scale=0.2)
    hand_r1 = _create_mock_hand(0.78, 0.8, scale=0.2)
    frames1 = [LandmarkFrame(pose=pose1, left_hand=None, right_hand=hand_r1) for _ in range(15)]
    tensor1 = LandmarkWindow(frames1).to_tensor()

    # Translated (+10.0, -5.0) and scaled by 2.5x
    pose2 = _create_mock_pose(ox=0.5 * 2.5 + 10.0, oy=0.5 * 2.5 - 5.0, scale=0.2 * 2.5)
    hand_r2 = _create_mock_hand(0.78 * 2.5 + 10.0, 0.8 * 2.5 - 5.0, scale=0.2 * 2.5)
    frames2 = [LandmarkFrame(pose=pose2, left_hand=None, right_hand=hand_r2) for _ in range(15)]
    tensor2 = LandmarkWindow(frames2).to_tensor()

    np.testing.assert_allclose(tensor1, tensor2, atol=1e-4)


def test_dominant_hand_canonicalisation_mirror_invariance() -> None:
    """Verify a sequence and its mirror produce the exact same tensor."""
    # Sequence where right hand moves (energy in right hand)
    pose_base = _create_mock_pose()
    frames_right_dominant = []
    frames_left_dominant = []

    for t in range(20):
        # Right hand moving horizontally
        hr = _create_mock_hand(0.78 + float(t) * 0.02, 0.8)
        frames_right_dominant.append(LandmarkFrame(pose=pose_base, left_hand=None, right_hand=hr))

        # Mirrored pose and left hand moving with exact mirror trajectory
        # In mirror: x -> -x + 1.0
        # Pose mirror: swap left/right pose pairs
        pose_mirrored = list(pose_base)
        for i in range(N_POSE):
            pose_mirrored[i] = (1.0 - pose_base[i][0], pose_base[i][1])
        # swap mouth L/R
        pose_mirrored[1], pose_mirrored[2] = pose_mirrored[2], pose_mirrored[1]
        # swap shoulder L/R
        pose_mirrored[3], pose_mirrored[4] = pose_mirrored[4], pose_mirrored[3]
        # swap elbow L/R
        pose_mirrored[5], pose_mirrored[6] = pose_mirrored[6], pose_mirrored[5]
        # swap wrist L/R
        pose_mirrored[7], pose_mirrored[8] = pose_mirrored[8], pose_mirrored[7]

        hl = tuple((1.0 - pt[0], pt[1]) for pt in hr)
        frames_left_dominant.append(
            LandmarkFrame(pose=tuple(pose_mirrored), left_hand=hl, right_hand=None)
        )

    tensor_right = LandmarkWindow(frames_right_dominant).to_tensor()
    tensor_left = LandmarkWindow(frames_left_dominant).to_tensor()

    np.testing.assert_allclose(tensor_right, tensor_left, atol=1e-4)
