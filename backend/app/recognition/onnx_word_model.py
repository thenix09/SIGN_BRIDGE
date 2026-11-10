"""ONNX runtime adapter for 30-word isolated ASL sign recognition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from backend.app.core.config import get_settings
from backend.app.core.exceptions import InvalidLandmarkError, ModelNotAvailableError
from backend.app.domain.gesture import Gesture
from backend.app.domain.sequence import LandmarkWindow
from backend.app.recognition.base import WordRecognizer


class ONNXWordRecognizer(WordRecognizer):
    """Production runtime WordRecognizer implementing ONNX inference for isolated ASL signs."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        vocab_path: Path | str | None = None,
        language: str = "ASL",
    ) -> None:
        """Initialize recognizer paths without loading weights immediately.

        Args:
            model_path: Optional path to word_model.onnx. Defaults to settings.word_model_path.
            vocab_path: Optional path to word_vocab.json. Defaults to settings.word_vocab_path.
            language: Language code. Defaults to 'ASL'.
        """
        settings = get_settings()
        self._model_path = Path(model_path or settings.word_model_path)
        self._vocab_path = Path(vocab_path or settings.word_vocab_path)
        self._language = language

        self._session: ort.InferenceSession | None = None
        self._vocabulary: tuple[str, ...] | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._is_loaded: bool = False

    @property
    def language(self) -> str:
        """Return the supported language code."""
        return self._language

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Return the closed vocabulary supported by this recognizer."""
        if self._vocabulary is None:
            self._load_vocabulary()
        assert self._vocabulary is not None
        return self._vocabulary

    @property
    def is_loaded(self) -> bool:
        """Return True if session and vocabulary have been successfully loaded."""
        return self._is_loaded

    def _load_vocabulary(self) -> None:
        """Load and strictly validate vocabulary metadata from JSON."""
        if not self._vocab_path.is_file():
            raise ModelNotAvailableError(
                f"Word vocabulary file not found at: {self._vocab_path}"
            )
        try:
            with open(self._vocab_path, encoding="utf-8") as f:
                data = json.load(f)

            vocab_list = data.get("vocabulary", [])
            if not isinstance(vocab_list, list) or len(vocab_list) != 30:
                raise ValueError(
                    f"Vocabulary must contain exactly 30 labels, got {len(vocab_list) if isinstance(vocab_list, list) else 0}."
                )
            if not all(isinstance(w, str) and w for w in vocab_list):
                raise ValueError("All vocabulary labels must be non-empty strings.")
            if len(set(vocab_list)) != 30:
                raise ValueError("Vocabulary contains duplicate labels.")

            label_to_id = data.get("label_to_id")
            if label_to_id is not None:
                if not isinstance(label_to_id, dict):
                    raise ValueError("label_to_id must be a dictionary.")
                expected_map = {word: i for i, word in enumerate(vocab_list)}
                if label_to_id != expected_map:
                    raise ValueError(
                        "Vocabulary label_to_id mapping disagrees with class index order."
                    )

            self._vocabulary = tuple(vocab_list)
        except Exception as exc:
            self._vocabulary = None
            raise ModelNotAvailableError(
                f"Failed to load word vocabulary from {self._vocab_path}: {exc}"
            ) from exc

    def load(self) -> None:
        """Load the ONNX model session and vocabulary. Safe to call multiple times."""
        if self._is_loaded:
            return

        try:
            # 1. Load vocabulary
            self._load_vocabulary()

            # 2. Verify model file exists
            if not self._model_path.is_file():
                raise ModelNotAvailableError(
                    f"Word recognition model file not found at: {self._model_path}"
                )

            # 3. Initialize ONNX runtime session
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if not inputs or not outputs:
                raise ValueError("ONNX model has no input or output nodes.")

            assert self._vocabulary is not None
            in_shape = inputs[0].shape
            if len(in_shape) != 3 or in_shape[1] != 32 or in_shape[2] != 104:
                raise ValueError(
                    f"ONNX input node shape must be (batch, 32, 104), got {in_shape}"
                )
            out_shape = outputs[0].shape
            if len(out_shape) != 2 or out_shape[1] != len(self._vocabulary):
                raise ValueError(
                    f"ONNX output node shape must be (batch, {len(self._vocabulary)}), got {out_shape}"
                )

            self._input_name = inputs[0].name
            self._output_name = outputs[0].name
            self._is_loaded = True
        except Exception as exc:
            self._is_loaded = False
            self._session = None
            self._vocabulary = None
            self._input_name = None
            self._output_name = None
            if isinstance(exc, ModelNotAvailableError):
                raise
            raise ModelNotAvailableError(
                f"Failed to initialize ONNX runtime session for {self._model_path}: {exc}"
            ) from exc

    def recognize(self, window: LandmarkWindow, top_k: int = 3) -> list[Gesture]:
        """Recognize isolated sign candidates from a LandmarkWindow.

        Args:
            window: Preprocessed LandmarkWindow.
            top_k: Number of top candidate predictions to return (ordered by confidence descending).

        Returns:
            List of top-k Gesture objects with label and confidence score.
        """
        if not self._is_loaded:
            self.load()

        assert self._session is not None
        assert self._vocabulary is not None
        assert self._input_name is not None
        assert self._output_name is not None

        if window is None or not isinstance(window, LandmarkWindow):
            raise InvalidLandmarkError(
                f"Expected LandmarkWindow instance, received: {type(window).__name__}"
            )

        try:
            tensor = window.to_tensor()
        except Exception as exc:
            raise InvalidLandmarkError(f"Failed to convert LandmarkWindow to tensor: {exc}") from exc

        if tensor.shape != (32, 104) or not np.isfinite(tensor).all():
            raise InvalidLandmarkError(
                f"Tensor failed validation: shape={tensor.shape}, finite={np.isfinite(tensor).all()}"
            )

        # Add batch dimension -> (1, 32, 104)
        input_data = np.expand_dims(tensor, axis=0).astype(np.float32)

        try:
            logits = self._session.run(
                [self._output_name],
                {self._input_name: input_data},
            )[0]
        except Exception as exc:
            raise ModelNotAvailableError(f"ONNX inference failure: {exc}") from exc

        if logits.ndim != 2 or logits.shape != (1, len(self._vocabulary)):
            raise ModelNotAvailableError(
                f"Unexpected ONNX output shape: expected (1, {len(self._vocabulary)}), got {logits.shape}"
            )
        if not np.isfinite(logits).all():
            raise ModelNotAvailableError("Non-finite values encountered in ONNX model logits.")

        if top_k <= 0:
            return []

        # Numerically stable softmax
        logits_row = logits[0]
        exp_logits = np.exp(logits_row - np.max(logits_row))
        probs = exp_logits / np.sum(exp_logits)

        # Sort indices by probability descending
        k = min(top_k, len(self._vocabulary))
        top_indices = np.argsort(probs)[::-1][:k]

        return [
            Gesture(label=self._vocabulary[idx], confidence=float(probs[idx]))
            for idx in top_indices
        ]
