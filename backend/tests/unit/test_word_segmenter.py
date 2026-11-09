"""Unit tests for SignSegmenter state tracking and window completion."""

from backend.app.domain.sequence import LandmarkFrame
from backend.app.recognition.segmenter import SegmenterState, SignSegmenter


def _create_frame(
    lw_x: float = 0.5,
    lw_y: float = 0.5,
    has_hand: bool = True,
) -> LandmarkFrame:
    """Helper to create dummy LandmarkFrame with specified left wrist coordinates."""
    pose = tuple((0.5, 0.5) for _ in range(9))
    if not has_hand:
        return LandmarkFrame(pose=pose, left_hand=None, right_hand=None)

    # 21 points for left hand
    hand_points = tuple((lw_x, lw_y) for _ in range(21))
    return LandmarkFrame(pose=pose, left_hand=hand_points, right_hand=None)


def test_segmenter_initial_state_idle() -> None:
    segmenter = SignSegmenter()
    assert segmenter.state == SegmenterState.IDLE

    # Frame with no hand keeps it in IDLE
    empty_frame = _create_frame(has_hand=False)
    state, window = segmenter.update(empty_frame)
    assert state == SegmenterState.IDLE
    assert window is None


def test_segmenter_starts_signing_on_motion() -> None:
    segmenter = SignSegmenter(motion_threshold=0.01)

    # First frame with hand triggers initial motion >= threshold
    frame1 = _create_frame(lw_x=0.2, lw_y=0.2)
    state, window = segmenter.update(frame1)
    assert state == SegmenterState.SIGNING
    assert window is None


def test_segmenter_discards_short_window() -> None:
    segmenter = SignSegmenter(min_frames=12, end_idle_frames=3, motion_threshold=0.01)

    # Start signing (5 moving frames)
    for i in range(5):
        frame = _create_frame(lw_x=0.2 + (i * 0.05), lw_y=0.2)
        segmenter.update(frame)

    # Then 3 idle frames (no hand)
    empty = _create_frame(has_hand=False)
    for _ in range(2):
        state, window = segmenter.update(empty)
        assert state == SegmenterState.SIGNING
        assert window is None

    # 3rd idle frame triggers end of short window -> discarded to IDLE
    state, window = segmenter.update(empty)
    assert state == SegmenterState.IDLE
    assert window is None


def test_segmenter_completes_valid_window_exactly_once() -> None:
    segmenter = SignSegmenter(min_frames=12, max_frames=50, end_idle_frames=4, motion_threshold=0.01)

    # 15 moving frames (>= min_frames)
    for i in range(15):
        frame = _create_frame(lw_x=0.1 + (i * 0.02), lw_y=0.1)
        state, window = segmenter.update(frame)
        assert state == SegmenterState.SIGNING
        assert window is None

    # 3 idle frames (stationary at position of frame 14)
    idle_frame = _create_frame(lw_x=0.1 + (14 * 0.02), lw_y=0.1)
    for _ in range(3):
        state, window = segmenter.update(idle_frame)
        assert state == SegmenterState.SIGNING
        assert window is None

    # 4th idle frame -> COMPLETE exactly once
    state, window = segmenter.update(idle_frame)
    assert state == SegmenterState.COMPLETE
    assert window is not None
    assert len(window.frames) == 19

    # Next update immediately returns to IDLE and window is None
    next_state, next_window = segmenter.update(_create_frame(has_hand=False))
    assert next_state == SegmenterState.IDLE
    assert next_window is None


def test_segmenter_completes_at_max_frames() -> None:
    segmenter = SignSegmenter(min_frames=5, max_frames=15, end_idle_frames=10)

    for i in range(14):
        frame = _create_frame(lw_x=0.1 + (i * 0.01), lw_y=0.1)
        state, window = segmenter.update(frame)
        assert state == SegmenterState.SIGNING
        assert window is None

    # 15th frame hits max_frames -> COMPLETE
    frame15 = _create_frame(lw_x=0.25, lw_y=0.1)
    state, window = segmenter.update(frame15)
    assert state == SegmenterState.COMPLETE
    assert window is not None
    assert len(window.frames) == 15


def test_segmenter_reset() -> None:
    segmenter = SignSegmenter()
    segmenter.update(_create_frame(lw_x=0.2, lw_y=0.2))
    assert segmenter.state == SegmenterState.SIGNING

    segmenter.reset()
    assert segmenter.state == SegmenterState.IDLE
