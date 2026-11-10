"""ONNX runtime implementation of WordRecognizer for isolated sign sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.exceptions import InvalidLandmarkError, ModelNotAvailableError
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
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._is_loaded = False

    @property
    def language(self) -> str:
        """Language code implemented by recognizer."""
        return self._language

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Closed vocabulary recognized by this model, without loading ONNX."""
        if not self._vocabulary:
            self._vocabulary = self._read_vocabulary()
        return tuple(self._vocabulary)

    @property
    def is_loaded(self) -> bool:
        """Whether both vocabulary and ONNX session are initialized."""
        return self._is_loaded

    def _read_vocabulary(self) -> list[str]:
        """Read and strictly validate the 30-class vocabulary and optional mappings."""
        if not self.vocab_path.is_file():
            raise ModelNotAvailableError(
                f"Word vocabulary file not found at {self.vocab_path}"
            )

        try:
            with self.vocab_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict):
                vocabulary = data.get("vocabulary")
                label_to_id = data.get("label_to_id")
                id_to_label = data.get("id_to_label")
            elif isinstance(data, list):
                vocabulary = data
                label_to_id = None
                id_to_label = None
            else:
                raise ValueError("Vocabulary JSON must contain an object or list.")

            if not isinstance(vocabulary, list) or len(vocabulary) != 30:
                raise ValueError("Vocabulary must contain exactly 30 labels.")

            if not all(isinstance(label, str) and label.strip() for label in vocabulary):
                raise ValueError("All vocabulary labels must be non-empty strings.")

            if len(set(vocabulary)) != 30:
                raise ValueError("Vocabulary contains duplicate labels.")

            expected_label_to_id = {
                label: index for index, label in enumerate(vocabulary)
            }
            expected_id_to_label = {
                str(index): label for index, label in enumerate(vocabulary)
            }

            if label_to_id is not None:
                if not isinstance(label_to_id, dict):
                    raise ValueError("label_to_id must be a dictionary.")
                if label_to_id != expected_label_to_id:
                    raise ValueError(
                        "label_to_id mapping disagrees with vocabulary ordering."
                    )

            if id_to_label is not None:
                if not isinstance(id_to_label, dict):
                    raise ValueError("id_to_label must be a dictionary.")
                if id_to_label != expected_id_to_label:
                    raise ValueError(
                        "id_to_label mapping disagrees with vocabulary ordering."
                    )

            return vocabulary

        except Exception as exc:
            raise ModelNotAvailableError(
                f"Failed to load word vocabulary from {self.vocab_path}: {exc}"
            ) from exc

    def load(self) -> None:
        """Load and validate vocabulary and ONNX session; safe to call repeatedly."""
        if self._is_loaded:
            return

        # Clear session state while preserving an independently loaded vocabulary.
        self._session = None
        self._input_name = None
        self._output_name = None
        self._is_loaded = False

        try:
            vocabulary = self._vocabulary or self._read_vocabulary()

            if not self.model_path.is_file():
                raise ModelNotAvailableError(
                    f"Word model file not found at {self.model_path}"
                )

            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1

            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

            inputs = session.get_inputs()
            outputs = session.get_outputs()

            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError(
                    "ONNX model must expose exactly one input and one output."
                )

            input_shape = inputs[0].shape
            output_shape = outputs[0].shape

            if (
                len(input_shape) != 3
                or input_shape[1] != 32
                or input_shape[2] != 104
            ):
                raise ValueError(
                    f"ONNX input must have shape (batch, 32, 104), got {input_shape}."
                )

            if len(output_shape) != 2 or output_shape[1] != len(vocabulary):
                raise ValueError(
                    f"ONNX output must have shape (batch, 30), got {output_shape}."
                )

            if input_shape[0] not in (1, "batch", None):
                raise ValueError(
                    f"ONNX input batch dimension must be 1 or dynamic, got {input_shape[0]}."
                )

            if output_shape[0] not in (1, "batch", None):
                raise ValueError(
                    f"ONNX output batch dimension must be 1 or dynamic, got {output_shape[0]}."
                )

            # Publish state only after every initialization check succeeds.
            self._session = session
            self._vocabulary = vocabulary
            self._input_name = inputs[0].name
            self._output_name = outputs[0].name
            self._is_loaded = True

        except ModelNotAvailableError:
            self._session = None
            self._vocabulary = []
            self._input_name = None
            self._output_name = None
            self._is_loaded = False
            raise

        except Exception as exc:
            self._session = None
            self._vocabulary = []
            self._input_name = None
            self._output_name = None
            self._is_loaded = False
            raise ModelNotAvailableError(
                f"Failed to initialize ONNX model from {self.model_path}: {exc}"
            ) from exc

    def recognize(
        self,
        window: LandmarkWindow,
        top_k: int = 3,
    ) -> list[Gesture]:
        """Recognize word candidates from a validated LandmarkWindow."""
        if window is None or not isinstance(window, LandmarkWindow):
            raise InvalidLandmarkError(
                f"Expected LandmarkWindow, received {type(window).__name__}."
            )

        if not self._is_loaded:
            self.load()

        try:
            tensor = np.asarray(window.to_tensor())

            if tensor.shape != (32, 104):
                raise InvalidLandmarkError(
                    f"Expected input tensor shape (32, 104), got {tensor.shape}."
                )

            if not np.isfinite(tensor).all():
                raise InvalidLandmarkError(
                    "Landmark tensor contains non-finite values."
                )

            input_data = np.expand_dims(tensor.astype(np.float32), axis=0)

            if not np.isfinite(input_data).all():
                raise InvalidLandmarkError(
                    "Float32 landmark tensor contains non-finite values."
                )

        except InvalidLandmarkError:
            raise
        except Exception as exc:
            raise InvalidLandmarkError(
                f"Failed to convert LandmarkWindow to a valid tensor: {exc}"
            ) from exc

        if top_k <= 0:
            return []

        assert self._session is not None
        assert self._input_name is not None
        assert self._output_name is not None

        try:
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: input_data},
            )
        except Exception as exc:
            raise ModelNotAvailableError(
                f"ONNX inference failed: {exc}"
            ) from exc

        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise ModelNotAvailableError(
                "ONNX model returned an invalid output collection."
            )

        logits = np.asarray(outputs[0])

        expected_shape = (1, len(self._vocabulary))
        if logits.shape != expected_shape:
            raise ModelNotAvailableError(
                f"Expected ONNX output shape {expected_shape}, got {logits.shape}."
            )

        if not np.isfinite(logits).all():
            raise ModelNotAvailableError(
                "ONNX model returned non-finite logits."
            )

        logits_row = logits[0]
        shifted_logits = logits_row - np.max(logits_row)
        exp_logits = np.exp(shifted_logits)
        denominator = np.sum(exp_logits)

        if not np.isfinite(denominator) or denominator <= 0:
            raise ModelNotAvailableError(
                "ONNX model produced invalid softmax probabilities."
            )

        probabilities = exp_logits / denominator

        if (
            not np.isfinite(probabilities).all()
            or not np.isclose(np.sum(probabilities), 1.0, atol=1e-6)
        ):
            raise ModelNotAvailableError(
                "ONNX model produced invalid normalized probabilities."
            )

        count = min(top_k, len(self._vocabulary))
        indices = np.argsort(probabilities)[::-1][:count]

        return [
            Gesture(
                label=self._vocabulary[index],
                confidence=float(probabilities[index]),
            )
            for index in indices
        ]
