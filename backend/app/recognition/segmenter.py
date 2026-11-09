"""Sign segmenter and motion state tracker for sequence word recognition."""

from __future__ import annotations

import math
from enum import StrEnum

from backend.app.domain.sequence import LandmarkFrame, LandmarkWindow


class SegmenterState(StrEnum):
    """Lifecycle state of the sign sequence segmenter."""

    IDLE = "IDLE"
    SIGNING = "SIGNING"
    COMPLETE = "COMPLETE"


class SignSegmenter:
    """Detects start and end of sign gestures from continuous landmark frames.

    Collects frames when a hand is present and moving; finishes after
    end_idle_frames frames without hands or without motion, or when reaching max_frames.
    Windows shorter than min_frames are discarded. A completed window is returned
    exactly once before returning to IDLE.
    """

    def __init__(
        self,
        min_frames: int = 12,
        max_frames: int = 90,
        end_idle_frames: int = 8,
        motion_threshold: float = 0.01,
    ) -> None:
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.end_idle_frames = end_idle_frames
        self.motion_threshold = motion_threshold

        self._state: SegmenterState = SegmenterState.IDLE
        self._buffer: list[LandmarkFrame] = []
        self._idle_count: int = 0
        self._prev_hand_frame: LandmarkFrame | None = None

    @property
    def state(self) -> SegmenterState:
        """Current lifecycle state."""
        return self._state

    def _has_hand(self, frame: LandmarkFrame | None) -> bool:
        """Check if at least one hand is detected in the frame."""
        if frame is None:
            return False
        return frame.left_hand is not None or frame.right_hand is not None

    def _compute_motion(self, frame: LandmarkFrame) -> float:
        """Compute maximum wrist displacement between current and previous frame with hands."""
        if self._prev_hand_frame is None:
            return self.motion_threshold

        motion_values: list[float] = []

        if frame.left_hand is not None and self._prev_hand_frame.left_hand is not None:
            curr_lw = frame.left_hand[0]
            prev_lw = self._prev_hand_frame.left_hand[0]
            motion_values.append(math.hypot(curr_lw[0] - prev_lw[0], curr_lw[1] - prev_lw[1]))

        if frame.right_hand is not None and self._prev_hand_frame.right_hand is not None:
            curr_rw = frame.right_hand[0]
            prev_rw = self._prev_hand_frame.right_hand[0]
            motion_values.append(math.hypot(curr_rw[0] - prev_rw[0], curr_rw[1] - prev_rw[1]))

        return max(motion_values) if motion_values else self.motion_threshold

    def update(self, frame: LandmarkFrame | None) -> tuple[SegmenterState, LandmarkWindow | None]:
        """Process incoming frame and return current state and completed window if ready.

        Args:
            frame: Incoming LandmarkFrame or None if no landmarks detected.

        Returns:
            Tuple of (current_state, completed_LandmarkWindow | None).
        """
        # If previously completed, transition back to IDLE
        if self._state == SegmenterState.COMPLETE:
            self._state = SegmenterState.IDLE
            self._buffer.clear()
            self._idle_count = 0
            self._prev_hand_frame = None

        hand_present = self._has_hand(frame)
        motion = self._compute_motion(frame) if (hand_present and frame is not None) else 0.0
        is_moving = hand_present and motion >= self.motion_threshold

        if self._state == SegmenterState.IDLE:
            if is_moving and frame is not None:
                self._state = SegmenterState.SIGNING
                self._buffer = [frame]
                self._idle_count = 0
                self._prev_hand_frame = frame
                return SegmenterState.SIGNING, None
            self._prev_hand_frame = None
            return SegmenterState.IDLE, None

        # State is SIGNING
        if frame is not None:
            self._buffer.append(frame)

        if not is_moving:
            self._idle_count += 1
        else:
            self._idle_count = 0

        if hand_present and frame is not None:
            self._prev_hand_frame = frame
        else:
            self._prev_hand_frame = None

        # Check completion condition: idle threshold or max length
        reached_idle = self._idle_count >= self.end_idle_frames
        reached_max = len(self._buffer) >= self.max_frames

        if reached_idle or reached_max:
            # Check if buffer satisfies minimum length requirement
            if len(self._buffer) >= self.min_frames:
                self._state = SegmenterState.COMPLETE
                completed_window = LandmarkWindow(self._buffer)
                self._buffer = []
                self._idle_count = 0
                self._prev_hand_frame = None
                return SegmenterState.COMPLETE, completed_window
            else:
                # Discard too short window and reset
                self._state = SegmenterState.IDLE
                self._buffer.clear()
                self._idle_count = 0
                self._prev_hand_frame = None
                return SegmenterState.IDLE, None

        return SegmenterState.SIGNING, None

    def reset(self) -> None:
        """Reset segmenter to initial IDLE state and clear all buffers."""
        self._state = SegmenterState.IDLE
        self._buffer.clear()
        self._idle_count = 0
        self._prev_hand_frame = None
