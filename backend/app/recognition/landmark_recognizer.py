"""Scikit-learn RandomForest landmark recognizer implementation."""

import threading
import weakref
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.domain.gesture import Gesture
from backend.app.domain.hand_frame import HandFrame
from backend.app.recognition.base import SignRecognizer


class _LandmarkModelResource:
    """Loaded model and class mapping shared by recognizers for one artifact."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model: Any = None
        self.classes: list[str] = []
        self.lock = threading.RLock()


_model_resources: weakref.WeakValueDictionary[str, _LandmarkModelResource] = (
    weakref.WeakValueDictionary()
)
_model_resources_lock = threading.Lock()


def _get_model_resource(model_path: Path) -> _LandmarkModelResource:
    key = str(model_path.resolve())
    with _model_resources_lock:
        resource = _model_resources.get(key)
        if resource is None:
            resource = _LandmarkModelResource(model_path)
            _model_resources[key] = resource
        return resource


class LandmarkRecognizer(SignRecognizer):
    """Recognizes letters from normalized 63-element feature vectors using scikit-learn."""

    def __init__(
        self, model_path: str | Path | None = None, threshold: float | None = None
    ) -> None:
        settings = get_settings()
        self.model_path = Path(model_path or settings.model_path)
        self.threshold = threshold if threshold is not None else settings.confidence_threshold
        self._resource = _get_model_resource(self.model_path)
        self._prob_ema: Any = None
        self._state_lock = threading.Lock()

    @property
    def language(self) -> str:
        """Language code implemented by subclass."""
        return "BASE"

    def load(self) -> None:
        """Load the serialized scikit-learn model."""
        with self._resource.lock:
            self._load_locked()

    def _load_locked(self) -> None:
        if self._resource.model is not None:
            return
        if not self.model_path.is_file():
            raise ModelNotAvailableError(f"Model file not found at {self.model_path}")
        try:
            model = joblib.load(self.model_path)
            classes = [str(c).upper() for c in getattr(model, "classes_", [])]
        except Exception as e:
            raise ModelNotAvailableError(f"Failed to load model from {self.model_path}: {e}")
        self._resource.model = model
        self._resource.classes = classes

    def recognize(self, frame: HandFrame) -> Gesture | None:
        """Recognize a letter gesture from a single HandFrame.

        Args:
            frame: A validated and canonicalized HandFrame.

        Returns:
            Gesture if top confidence >= threshold, else None.
        """
        features = frame.to_feature_vector()
        x_input = np.array([features], dtype=np.float32)

        with self._state_lock:
            with self._resource.lock:
                self._load_locked()
                model = self._resource.model
                classes = self._resource.classes

                if hasattr(model, "predict_proba"):
                    raw_probs = model.predict_proba(x_input)[0]
                    if self._prob_ema is None or len(self._prob_ema) != len(raw_probs):
                        self._prob_ema = raw_probs
                    else:
                        # Exponential moving average smoothing (alpha=0.35 stabilizes frame jitter)
                        self._prob_ema = 0.35 * raw_probs + 0.65 * self._prob_ema

                    probs = self._prob_ema
                    max_idx = int(np.argmax(probs))
                    confidence = float(probs[max_idx])
                    label = (
                        classes[max_idx]
                        if max_idx < len(classes)
                        else str(model.predict(x_input)[0])
                    )
                else:
                    pred = model.predict(x_input)[0]
                    label = str(pred).upper()
                    confidence = 1.0

        if confidence < self.threshold:
            return None

        return Gesture(label=label, confidence=round(confidence, 4))
