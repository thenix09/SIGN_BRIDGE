"""Unit tests for letter domain models: Landmark, HandFrame, canonicalise_landmarks, Sample, Translation."""

import math
from datetime import UTC, datetime

import pytest

from backend.app.core.exceptions import InvalidLandmarkError
from backend.app.domain.hand_frame import HandFrame, canonicalise_landmarks
from backend.app.domain.landmark import Landmark
from backend.app.domain.sample import Sample
from backend.app.domain.translation import Translation


def test_landmark_valid() -> None:
    """Verify Landmark stores valid coordinates."""
    lm = Landmark(x=0.1, y=0.2, z=0.3)
    assert lm.x == 0.1
    assert lm.y == 0.2
    assert lm.z == 0.3


def test_landmark_rejects_nan_and_inf() -> None:
    """Verify Landmark rejects NaN and Infinity with InvalidLandmarkError."""
    with pytest.raises(InvalidLandmarkError):
        Landmark(x=float("nan"), y=0.0, z=0.0)

    with pytest.raises(InvalidLandmarkError):
        Landmark(x=0.0, y=float("inf"), z=0.0)

    with pytest.raises(InvalidLandmarkError):
        Landmark(x=0.0, y=0.0, z=float("-inf"))


def test_hand_frame_rejects_wrong_length() -> None:
    """Verify HandFrame raises InvalidLandmarkError if landmark count is not 21."""
    short_list = [Landmark(x=0.0, y=0.0, z=0.0) for _ in range(20)]
    with pytest.raises(InvalidLandmarkError):
        HandFrame(short_list)

    long_list = [Landmark(x=0.0, y=0.0, z=0.0) for _ in range(22)]
    with pytest.raises(InvalidLandmarkError):
        HandFrame(long_list)


def test_hand_frame_from_raw() -> None:
    """Verify HandFrame.from_raw parses 21x3 raw lists."""
    raw = [[float(i), float(i + 1), float(i + 2)] for i in range(21)]
    frame = HandFrame.from_raw(raw)
    assert len(frame.landmarks) == 21
    assert frame.landmarks[0].x == 0.0
    assert frame.landmarks[20].z == 22.0


def test_hand_frame_from_raw_invalid() -> None:
    """Verify from_raw raises InvalidLandmarkError on invalid coordinates or counts."""
    with pytest.raises(InvalidLandmarkError):
        HandFrame.from_raw([[0.0, 0.0, 0.0]] * 15)

    with pytest.raises(InvalidLandmarkError):
        HandFrame.from_raw([[0.0, 0.0]] * 21)  # needs 3 coordinates


def test_hand_frame_translation_and_scale_invariance() -> None:
    """Verify to_feature_vector is translation and scale invariant and deterministic."""
    # Base hand shape
    raw1 = [[float(i) * 0.02, float(i) * 0.03, float(i) * 0.01] for i in range(21)]
    frame1 = HandFrame.from_raw(raw1)
    feat1 = frame1.to_feature_vector()
    assert len(feat1) == 63

    # Translated by (+5.0, -3.0, +2.0)
    raw2 = [[pt[0] + 5.0, pt[1] - 3.0, pt[2] + 2.0] for pt in raw1]
    frame2 = HandFrame.from_raw(raw2)
    feat2 = frame2.to_feature_vector()

    # Scaled by 3.5x
    raw3 = [[pt[0] * 3.5, pt[1] * 3.5, pt[2] * 3.5] for pt in raw1]
    frame3 = HandFrame.from_raw(raw3)
    feat3 = frame3.to_feature_vector()

    # Determinism
    assert feat1 == frame1.to_feature_vector()

    # Invariance check
    for f1, f2, f3 in zip(feat1, feat2, feat3):
        assert math.isclose(f1, f2, abs_tol=1e-5)
        assert math.isclose(f1, f3, abs_tol=1e-5)


def test_canonicalise_landmarks_mirrors_opposite_hand() -> None:
    """Verify canonicalise_landmarks mirrors x coordinates when handedness != canonical."""
    raw = [
        [0.2, 0.5, 0.1],
        [0.8, 0.4, 0.3],
    ]
    canonical = "Right"

    # Same handedness -> untouched
    res_right = canonicalise_landmarks(raw, "Right", canonical)
    assert res_right[0] == [0.2, 0.5, 0.1]
    assert res_right[1] == [0.8, 0.4, 0.3]

    # Opposite handedness -> mirrored x (1 - x)
    res_left = canonicalise_landmarks(raw, "Left", canonical)
    assert math.isclose(res_left[0][0], 0.8)
    assert math.isclose(res_left[0][1], 0.5)
    assert math.isclose(res_left[1][0], 0.2)
    assert math.isclose(res_left[1][1], 0.4)


def test_left_hand_mirrored_produces_identical_feature_vector() -> None:
    """Verify a left-hand input is mirrored to the same vector as the equivalent right-hand input."""
    # Right hand coordinates
    right_raw = [[float(i) * 0.03, float(i) * 0.02, 0.01] for i in range(21)]
    # Equivalent left hand: same y and z, but mirrored x (1 - x)
    left_raw = [[1.0 - pt[0], pt[1], pt[2]] for pt in right_raw]

    # Canonicalize both with canonical = 'Right'
    canon_right = canonicalise_landmarks(right_raw, "Right", "Right")
    canon_left = canonicalise_landmarks(left_raw, "Left", "Right")

    frame_right = HandFrame.from_raw(canon_right)
    frame_left = HandFrame.from_raw(canon_left)

    feat_right = frame_right.to_feature_vector()
    feat_left = frame_left.to_feature_vector()

    for r_val, l_val in zip(feat_right, feat_left):
        assert math.isclose(r_val, l_val, abs_tol=1e-5)


def test_sample_dataclass_validation() -> None:
    """Verify Sample dataclass requires 63 finite features."""
    valid_features = tuple([0.1] * 63)
    s = Sample(label="A", group="img01", features=valid_features)
    assert s.label == "A"
    assert s.group == "img01"
    assert len(s.features) == 63

    # Reject wrong length
    with pytest.raises(InvalidLandmarkError):
        Sample(label="A", group="img01", features=tuple([0.1] * 62))

    # Reject NaN
    bad_features = tuple([float("nan")] + [0.1] * 62)
    with pytest.raises(InvalidLandmarkError):
        Sample(label="A", group="img01", features=bad_features)


def test_translation_dataclass() -> None:
    """Verify Translation dataclass fields."""
    now = datetime.now(UTC)
    t = Translation(
        id=1,
        direction="sign_to_text",
        language="ASL",
        content="HELLO",
        created_at=now,
    )
    assert t.id == 1
    assert t.direction == "sign_to_text"
    assert t.language == "ASL"
    assert t.content == "HELLO"
    assert t.created_at == now
