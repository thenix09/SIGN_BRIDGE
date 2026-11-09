"""Scikit-learn RandomForest landmark recognizer implementation."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.domain.gesture import Gesture
from backend.app.domain.hand_frame import HandFrame
from backend.app.recognition.base import SignRecognizer


class LandmarkRecognizer(SignRecognizer):
    """Recognizes letters from normalized 63-element feature vectors using scikit-learn."""

    def __init__(
        self, model_path: str | Path | None = None, threshold: float | None = None
    ) -> None:
        settings = get_settings()
        self.model_path = Path(model_path or settings.model_path)
        self.threshold = threshold if threshold is not None else settings.confidence_threshold
        self._model: Any = None
        self._classes: list[str] = []
        self._prob_ema: Any = None

    @property
    def language(self) -> str:
        """Language code implemented by subclass."""
        return "BASE"

    def load(self) -> None:
        """Load the serialized scikit-learn model."""
        if not self.model_path.is_file():
            raise ModelNotAvailableError(f"Model file not found at {self.model_path}")
        try:
            self._model = joblib.load(self.model_path)
            self._classes = [str(c).upper() for c in getattr(self._model, "classes_", [])]
        except Exception as e:
            raise ModelNotAvailableError(f"Failed to load model from {self.model_path}: {e}")

    def recognize(self, frame: HandFrame) -> Gesture | None:
        """Recognize a letter gesture from a single HandFrame.

        Args:
            frame: A validated and canonicalized HandFrame.

        Returns:
            Gesture if top confidence >= threshold, else None.
        """
        if self._model is None:
            self.load()

        features = frame.to_feature_vector()
        x_input = np.array([features], dtype=np.float32)

        if hasattr(self._model, "predict_proba"):
            raw_probs = self._model.predict_proba(x_input)[0]
            if not hasattr(self, "_prob_ema") or self._prob_ema is None or len(self._prob_ema) != len(raw_probs):
                self._prob_ema = raw_probs
            else:
                # Exponential moving average smoothing (alpha=0.35 stabilizes frame jitter)
                self._prob_ema = 0.35 * raw_probs + 0.65 * self._prob_ema

            probs = self._prob_ema
            max_idx = int(np.argmax(probs))
            confidence = float(probs[max_idx])
            label = (
                self._classes[max_idx]
                if max_idx < len(self._classes)
                else str(self._model.predict(x_input)[0])
            )
        else:
            pred = self._model.predict(x_input)[0]
            label = str(pred).upper()
            confidence = 1.0

        if confidence < self.threshold:
            return None

        return Gesture(label=label, confidence=round(confidence, 4))
