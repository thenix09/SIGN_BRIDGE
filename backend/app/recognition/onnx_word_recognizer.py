"""ONNX runtime implementation of WordRecognizer for isolated sign sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.domain.gesture import Gesture
from backend.app.domain.sequence import LandmarkWindow
from backend.app.recognition.base import WordRecognizer


class OnnxWordRecognizer(WordRecognizer):
    """Recognizes sequence words from a LandmarkWindow using an exported ONNX model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        language: str = "ASL",
    ) -> None:
        settings = get_settings()
        self.model_path = Path(model_path or settings.word_model_path)
        self.vocab_path = Path(vocab_path or settings.word_vocab_path)
        self._language = language
        self._session: Any = None
        self._vocabulary: list[str] = []

    @property
    def language(self) -> str:
        """Language code implemented by recognizer."""
        return self._language

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Closed vocabulary recognized by this model."""
        return tuple(self._vocabulary)

    def load(self) -> None:
        """Load vocabulary and instantiate the ONNX inference session."""
        if not self.model_path.is_file():
            raise ModelNotAvailableError(
                f"Word model file not found at {self.model_path}"
            )
        if not self.vocab_path.is_file():
            raise ModelNotAvailableError(
                f"Word vocabulary file not found at {self.vocab_path}"
            )

        try:
            with self.vocab_path.open("r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            if isinstance(vocab_data, dict):
                self._vocabulary = list(vocab_data.get("vocabulary", []))
            elif isinstance(vocab_data, list):
                self._vocabulary = [str(item) for item in vocab_data]
            else:
                self._vocabulary = []
        except Exception as e:
            raise ModelNotAvailableError(f"Failed to read vocabulary from {self.vocab_path}: {e}")

        try:
            import onnxruntime as ort

            # Use CPUExecutionProvider for deterministic inference
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            raise ModelNotAvailableError(f"Failed to load ONNX model from {self.model_path}: {e}")

    def recognize(self, window: LandmarkWindow, top_k: int = 3) -> list[Gesture]:
        """Recognize sign candidates from a LandmarkWindow.

        Args:
            window: Preprocessed LandmarkWindow.
            top_k: Maximum candidate predictions to return.

        Returns:
            Top-k gestures ordered by confidence descending; never raises for low confidence.
        """
        if self._session is None or not self._vocabulary:
            self.load()

        tensor = window.to_tensor()  # Shape: (32, 104)
        batch = np.expand_dims(tensor, axis=0).astype(np.float32)  # Shape: (1, 32, 104)

        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name

        outputs = self._session.run([output_name], {input_name: batch})[0]
        logits = outputs[0]  # Shape: (num_classes,)

        # Numerically stable softmax
        shift_logits = logits - np.max(logits)
        exp_logits = np.exp(shift_logits)
        probs = exp_logits / np.sum(exp_logits)

        # Get top-k indices sorted descending
        k = min(top_k, len(probs), len(self._vocabulary))
        top_indices = np.argsort(probs)[::-1][:k]

        return [
            Gesture(
                label=self._vocabulary[idx],
                confidence=float(probs[idx]),
            )
            for idx in top_indices
        ]
