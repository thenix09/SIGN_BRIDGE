"""Stability filter for fingerspelling gesture stream smoothing."""

from enum import StrEnum

from backend.app.domain.gesture import Gesture


class RecognizerState(StrEnum):
    """Recognition state lifecycle."""

    IDLE = "IDLE"
    DETECTING = "DETECTING"
    STABILISING = "STABILISING"
    CONFIRMED = "CONFIRMED"


class StabilityFilter:
    """Stateful filter requiring consecutive frames of stable recognition before letter emission.

    Rules:
        - Requires required_frames (default 8) consecutive frames of the same letter.
        - Emits a confirmed letter exactly once.
        - After emission, enters cooldown_frames (default 10) before the same letter
          can be confirmed again, or requires hand to change/disappear.
    """

    def __init__(self, required_frames: int = 8, cooldown_frames: int = 10) -> None:
        self.required_frames = required_frames
        self.cooldown_frames = cooldown_frames

        self._candidate_letter: str | None = None
        self._streak_count: int = 0
        self._cooldown_count: int = 0
        self._last_confirmed_letter: str | None = None
        self._state: RecognizerState = RecognizerState.IDLE

    @property
    def state(self) -> RecognizerState:
        """Return current recognizer state."""
        return self._state

    def reset(self) -> None:
        """Reset internal filter tracking state."""
        self._candidate_letter = None
        self._streak_count = 0
        self._cooldown_count = 0
        self._last_confirmed_letter = None
        self._state = RecognizerState.IDLE

    def update(self, gesture: Gesture | None) -> tuple[RecognizerState, str | None]:
        """Update filter state with latest recognized gesture candidate.

        Args:
            gesture: Detected gesture or None if no hand/low confidence.

        Returns:
            Tuple of (current_state, confirmed_letter_if_newly_confirmed).
        """
        # Decrement cooldown counter if active
        if self._cooldown_count > 0:
            self._cooldown_count -= 1

        if gesture is None or not gesture.label:
            # Hand disappeared or confidence below threshold
            self._candidate_letter = None
            self._streak_count = 0
            self._last_confirmed_letter = None
            self._state = RecognizerState.IDLE
            return self._state, None

        letter = gesture.label.upper()

        if letter != self._candidate_letter:
            # Different letter candidate detected
            self._candidate_letter = letter
            self._streak_count = 1
            self._state = RecognizerState.DETECTING
            return self._state, None

        # Same candidate continues
        self._streak_count += 1

        if self._streak_count < self.required_frames:
            self._state = RecognizerState.STABILISING
            return self._state, None

        # Streak threshold reached
        if self._streak_count == self.required_frames:
            # Check if this same letter is still in cooldown
            if letter == self._last_confirmed_letter and self._cooldown_count > 0:
                self._state = RecognizerState.STABILISING
                return self._state, None

            # Confirm letter exactly once
            self._state = RecognizerState.CONFIRMED
            self._last_confirmed_letter = letter
            self._cooldown_count = self.cooldown_frames
            return self._state, letter

        # streak_count > required_frames: still holding same letter
        self._state = RecognizerState.STABILISING
        return self._state, None
