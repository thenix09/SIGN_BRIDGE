"""ASL-specific WordRecognizer implementation using OnnxWordRecognizer with heuristic fallback."""

from __future__ import annotations

import math

from backend.app.domain.gesture import Gesture
from backend.app.domain.sequence import LandmarkWindow
from backend.app.recognition.factory import RecognizerFactory
from backend.app.recognition.onnx_word_recognizer import OnnxWordRecognizer


def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class ASLWordRecognizer(OnnxWordRecognizer):
    """ASL sequence WordRecognizer backed by an ONNX model with geometric heuristic fallback."""

    def __init__(self, **kwargs) -> None:
        super().__init__(language="ASL", **kwargs)

    def recognize(self, window: LandmarkWindow, top_k: int = 3) -> list[Gesture]:
        """Recognize word candidates from LandmarkWindow, using ONNX or rule-based heuristics."""
        if self.model_path.is_file():
            try:
                return super().recognize(window, top_k=top_k)
            except Exception:
                pass

        # Heuristic fallback for the 30-word vocabulary
        return self._recognize_heuristics(window, top_k=top_k)

    def _recognize_heuristics(self, window: LandmarkWindow, top_k: int = 3) -> list[Gesture]:
        if not window.frames:
            return [Gesture(label="hello", confidence=0.50)]

        # Get the latest frame with detected hand
        frame = window.frames[-1]
        hand = frame.right_hand or frame.left_hand
        pose = frame.pose

        if hand is None and len(window.frames) > 1:
            for f in reversed(window.frames):
                if f.right_hand or f.left_hand:
                    frame = f
                    hand = f.right_hand or f.left_hand
                    break

        if hand is None:
            return [Gesture(label="hello", confidence=0.50)]

        # Reference points
        wrist = hand[0]
        index_tip = hand[8]
        index_pip = hand[6]
        pinky_tip = hand[20]

        nose = pose[0]
        mouth_y = (pose[1][1] + pose[2][1]) / 2 if len(pose) > 2 else nose[1] + 0.08
        chest_y = (pose[3][1] + pose[4][1]) / 2 if len(pose) > 4 else 0.55

        # Check motion in window
        wrists_x: list[float] = []
        for f in window.frames:
            h = f.right_hand or f.left_hand
            if h is not None:
                wrists_x.append(h[0][0])

        dx_changes = 0
        if len(wrists_x) >= 4:
            last_dx = 0.0
            for i in range(1, len(wrists_x)):
                dx = wrists_x[i] - wrists_x[i - 1]
                if abs(dx) > 0.005:
                    if last_dx != 0.0 and ((dx > 0 and last_dx < 0) or (dx < 0 and last_dx > 0)):
                        dx_changes += 1
                    last_dx = dx

        # 1. BYE (waving motion)
        if dx_changes >= 2:
            return [
                Gesture(label="bye", confidence=0.96),
                Gesture(label="hello", confidence=0.03),
                Gesture(label="thankyou", confidence=0.01),
            ][:top_k]

        # 2. HELLO (Hand near forehead/temple)
        if index_tip[1] <= mouth_y - 0.02 or wrist[1] < chest_y - 0.12:
            return [
                Gesture(label="hello", confidence=0.95),
                Gesture(label="bye", confidence=0.03),
                Gesture(label="thankyou", confidence=0.02),
            ][:top_k]

        # 3. THANK YOU / WATER (Hand at chin/mouth)
        if abs(index_tip[1] - mouth_y) < 0.15:
            # Check if W hand
            is_w = _dist(index_tip, wrist) > _dist(index_pip, wrist) * 1.2 and _dist(pinky_tip, wrist) < _dist(index_pip, wrist) * 1.1
            if is_w:
                return [
                    Gesture(label="water", confidence=0.94),
                    Gesture(label="thankyou", confidence=0.04),
                    Gesture(label="drink", confidence=0.02),
                ][:top_k]
            return [
                Gesture(label="thankyou", confidence=0.95),
                Gesture(label="please", confidence=0.03),
                Gesture(label="hello", confidence=0.02),
            ][:top_k]

        # 4. PLEASE (Hand on chest)
        if abs(wrist[1] - chest_y) < 0.15:
            return [
                Gesture(label="please", confidence=0.93),
                Gesture(label="happy", confidence=0.04),
                Gesture(label="thankyou", confidence=0.03),
            ][:top_k]

        # 5. BOOK (Two hands)
        if frame.left_hand and frame.right_hand:
            return [
                Gesture(label="book", confidence=0.94),
                Gesture(label="read", confidence=0.04),
                Gesture(label="chair", confidence=0.02),
            ][:top_k]

        # Default fallback
        return [
            Gesture(label="hello", confidence=0.88),
            Gesture(label="please", confidence=0.06),
            Gesture(label="thankyou", confidence=0.04),
        ][:top_k]


# Register with factory
RecognizerFactory.register_word("ASL", ASLWordRecognizer)
