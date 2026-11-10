"""Verify the provisioned PyTorch and ONNX word-model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from google.protobuf.message import DecodeError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.recognition.word_model import WordSignModel  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "backend" / "models" / "word_model.pt"
DEFAULT_ONNX = ROOT / "backend" / "models" / "word_model.onnx"
DEFAULT_VOCABULARY = ROOT / "backend" / "models" / "word_vocab.json"
DEFAULT_SAMPLE_DATA = ROOT / "backend" / "data" / "processed" / "word_dataset.npz"
DEFAULT_METADATA = ROOT / "backend" / "data" / "processed" / "meta.json"
PARITY_TOLERANCE = 1e-4
NUM_CLASSES = 30
INPUT_SHAPE = (32, 104)
SAMPLE_BATCH_SIZE = 4


class ArtifactVerificationError(RuntimeError):
    """Raised when a provisioned model artifact fails verification."""


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ArtifactVerificationError(f"{description} not found: {path}")


def _read_vocabulary(path: Path) -> tuple[list[str], dict[str, int], dict[str, str]]:
    try:
        with path.open(encoding="utf-8") as file:
            data: Any = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactVerificationError(f"Could not read vocabulary {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ArtifactVerificationError("Vocabulary JSON must be an object.")
    vocabulary = data.get("vocabulary")
    if (
        not isinstance(vocabulary, list)
        or len(vocabulary) != NUM_CLASSES
        or not all(isinstance(label, str) and label for label in vocabulary)
        or len(set(vocabulary)) != NUM_CLASSES
    ):
        raise ArtifactVerificationError(
            f"Vocabulary must contain exactly {NUM_CLASSES} unique, non-empty labels."
        )

    expected_label_to_id = {label: index for index, label in enumerate(vocabulary)}
    expected_id_to_label = {str(index): label for index, label in enumerate(vocabulary)}
    if data.get("label_to_id") != expected_label_to_id:
        raise ArtifactVerificationError("Vocabulary label_to_id does not match class order.")
    if data.get("id_to_label") != expected_id_to_label:
        raise ArtifactVerificationError("Vocabulary id_to_label does not match class order.")
    return vocabulary, expected_label_to_id, expected_id_to_label


def _load_sample(sample_data_path: Path) -> tuple[np.ndarray, str]:
    if sample_data_path.is_file():
        try:
            with np.load(sample_data_path, allow_pickle=False) as archive:
                if "X" not in archive:
                    raise ArtifactVerificationError(
                        f"Processed data has no X tensor array: {sample_data_path}"
                    )
                features = archive["X"]
                if features.ndim != 3 or features.shape[1:] != INPUT_SHAPE:
                    raise ArtifactVerificationError(
                        f"Processed X must have shape (N, {INPUT_SHAPE[0]}, "
                        f"{INPUT_SHAPE[1]}), got {features.shape}."
                    )
                if features.shape[0] == 0 or features.dtype != np.float32:
                    raise ArtifactVerificationError(
                        "Processed X must contain float32 samples."
                    )
                if not np.isfinite(features).all():
                    raise ArtifactVerificationError("Processed X contains non-finite values.")

                indices = np.arange(features.shape[0])
                if "split" in archive:
                    splits = archive["split"].astype(str)
                    test_indices = np.flatnonzero(splits == "test")
                    if test_indices.size:
                        indices = test_indices
                sample = np.ascontiguousarray(
                    features[indices[:SAMPLE_BATCH_SIZE]], dtype=np.float32
                )
        except (OSError, ValueError) as exc:
            if isinstance(exc, ArtifactVerificationError):
                raise
            raise ArtifactVerificationError(
                f"Could not read processed sample data {sample_data_path}: {exc}"
            ) from exc
        return sample, f"processed data: {sample_data_path}"

    sample = np.random.default_rng(42).standard_normal(
        (SAMPLE_BATCH_SIZE, *INPUT_SHAPE), dtype=np.float32
    )
    return sample, "deterministic synthetic input (processed tensors unavailable)"


def _validate_batch_dimension(shape: list[Any], description: str) -> None:
    if not shape or not (shape[0] is None or isinstance(shape[0], str)):
        raise ArtifactVerificationError(
            f"{description} must have a dynamic batch dimension, got {shape}."
        )


def verify_word_model_artifacts(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    onnx_path: Path = DEFAULT_ONNX,
    vocabulary_path: Path = DEFAULT_VOCABULARY,
    sample_data_path: Path = DEFAULT_SAMPLE_DATA,
    metadata_path: Path = DEFAULT_METADATA,
    tolerance: float = PARITY_TOLERANCE,
) -> list[str]:
    """Validate checkpoint, vocabulary, ONNX runtime, and numerical inference parity."""
    for path, description in (
        (checkpoint_path, "PyTorch checkpoint"),
        (onnx_path, "ONNX model"),
        (vocabulary_path, "Vocabulary"),
    ):
        _require_file(path, description)
    if tolerance <= 0:
        raise ArtifactVerificationError("Parity tolerance must be positive.")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise ArtifactVerificationError(
            f"Could not load PyTorch checkpoint {checkpoint_path}: {exc}"
        ) from exc
    required_keys = {"model_state_dict", "vocabulary", "num_classes", "total_params"}
    if not isinstance(checkpoint, dict):
        raise ArtifactVerificationError("Checkpoint must contain a mapping of model metadata.")
    if not required_keys.issubset(checkpoint):
        raise ArtifactVerificationError(
            f"Checkpoint is missing required keys: {sorted(required_keys - set(checkpoint))}."
        )

    vocabulary, label_to_id, id_to_label = _read_vocabulary(vocabulary_path)
    if checkpoint["vocabulary"] != vocabulary:
        raise ArtifactVerificationError(
            "Checkpoint vocabulary ordering does not match word_vocab.json."
        )
    if checkpoint["num_classes"] != NUM_CLASSES:
        raise ArtifactVerificationError(
            f"Checkpoint num_classes must be {NUM_CLASSES}, got {checkpoint['num_classes']}."
        )

    model = WordSignModel(
        in_features=104,
        num_classes=NUM_CLASSES,
        cnn_channels=(128, 128),
        gru_hidden_size=128,
        num_gru_layers=2,
        dropout=0.3,
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ArtifactVerificationError(f"Checkpoint weights do not match WordSignModel: {exc}") from exc
    model.eval()
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if checkpoint["total_params"] != parameter_count:
        raise ArtifactVerificationError(
            f"Checkpoint declares {checkpoint['total_params']} parameters; model has "
            f"{parameter_count}."
        )

    lines = [
        f"PASS checkpoint loaded; epoch={checkpoint.get('epoch', 'unknown')}; "
        f"trainable_parameters={parameter_count}",
        f"PASS vocabulary has {len(vocabulary)} unique labels and consistent class-index maps",
        "PASS checkpoint vocabulary ordering matches word_vocab.json",
    ]

    if metadata_path.is_file():
        try:
            with metadata_path.open(encoding="utf-8") as file:
                metadata: Any = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactVerificationError(
                f"Could not read processed metadata {metadata_path}: {exc}"
            ) from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("vocabulary") != vocabulary
            or metadata.get("label_to_id") != label_to_id
            or metadata.get("id_to_label") != id_to_label
        ):
            raise ArtifactVerificationError(
                f"Processed metadata class ordering does not match {vocabulary_path}."
            )
        lines.append("PASS processed metadata class ordering matches vocabulary")
    else:
        lines.append(f"INFO processed metadata unavailable; skipped mapping check: {metadata_path}")

    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model, full_check=True)
    except (OSError, DecodeError, onnx.checker.ValidationError, ValueError) as exc:
        raise ArtifactVerificationError(f"ONNX validation failed for {onnx_path}: {exc}") from exc

    metadata_properties = {
        property_.key: property_.value for property_ in onnx_model.metadata_props
    }
    for key, expected in (
        ("vocabulary", vocabulary),
        ("label_to_id", label_to_id),
    ):
        if key not in metadata_properties:
            raise ArtifactVerificationError(f"ONNX model is missing {key} class-order metadata.")
        try:
            if json.loads(metadata_properties[key]) != expected:
                raise ArtifactVerificationError(f"ONNX {key} metadata does not match vocabulary.")
        except json.JSONDecodeError as exc:
            raise ArtifactVerificationError(f"ONNX {key} metadata is not valid JSON.") from exc
    lines.append("PASS ONNX checker")

    try:
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactVerificationError(f"ONNX Runtime could not load model: {exc}") from exc
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ArtifactVerificationError(
            f"Expected CPU-only ONNX Runtime, got {session.get_providers()}."
        )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ArtifactVerificationError(
            f"Expected one ONNX input and output, got {len(inputs)} and {len(outputs)}."
        )
    input_node, output_node = inputs[0], outputs[0]
    if input_node.name != "landmarks" or output_node.name != "logits":
        raise ArtifactVerificationError(
            f"Unexpected ONNX names: input={input_node.name}, output={output_node.name}."
        )
    if input_node.type != "tensor(float)" or output_node.type != "tensor(float)":
        raise ArtifactVerificationError(
            "ONNX input and output must use float32 tensors; "
            f"got {input_node.type} and {output_node.type}."
        )
    if len(input_node.shape) != 3 or input_node.shape[1:] != list(INPUT_SHAPE):
        raise ArtifactVerificationError(
            f"ONNX input must have shape (batch, {INPUT_SHAPE[0]}, {INPUT_SHAPE[1]}), "
            f"got {input_node.shape}."
        )
    if len(output_node.shape) != 2 or output_node.shape[1] != NUM_CLASSES:
        raise ArtifactVerificationError(
            f"ONNX output must have shape (batch, {NUM_CLASSES}), got {output_node.shape}."
        )
    _validate_batch_dimension(input_node.shape, "ONNX input")
    _validate_batch_dimension(output_node.shape, "ONNX output")
    lines.append(
        f"PASS ONNX Runtime CPU session; input={input_node.shape}; output={output_node.shape}"
    )

    sample, sample_source = _load_sample(sample_data_path)
    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(sample)).cpu().numpy()
        torch_probabilities = torch.softmax(torch.from_numpy(torch_logits), dim=-1).numpy()
    try:
        onnx_logits = session.run(
            [output_node.name], {input_node.name: sample}
        )[0]
    except (RuntimeError, ValueError) as exc:
        raise ArtifactVerificationError(f"ONNX inference failed: {exc}") from exc
    if torch_logits.shape != (len(sample), NUM_CLASSES):
        raise ArtifactVerificationError(f"PyTorch returned unexpected logits shape {torch_logits.shape}.")
    if onnx_logits.shape != torch_logits.shape:
        raise ArtifactVerificationError(
            f"ONNX logits shape {onnx_logits.shape} does not match PyTorch {torch_logits.shape}."
        )
    if not np.isfinite(torch_logits).all() or not np.isfinite(onnx_logits).all():
        raise ArtifactVerificationError("Model inference produced non-finite logits.")

    shifted_logits = onnx_logits - np.max(onnx_logits, axis=-1, keepdims=True)
    onnx_exp = np.exp(shifted_logits)
    onnx_probabilities = onnx_exp / np.sum(onnx_exp, axis=-1, keepdims=True)
    if not np.isfinite(torch_probabilities).all() or not np.isfinite(onnx_probabilities).all():
        raise ArtifactVerificationError("Softmax produced non-finite probabilities.")
    if not np.allclose(torch_probabilities.sum(axis=-1), 1.0, atol=tolerance, rtol=0):
        raise ArtifactVerificationError("PyTorch softmax probabilities are not normalized.")
    if not np.allclose(onnx_probabilities.sum(axis=-1), 1.0, atol=tolerance, rtol=0):
        raise ArtifactVerificationError("ONNX softmax probabilities are not normalized.")

    max_difference = float(np.max(np.abs(torch_logits - onnx_logits)))
    if max_difference > tolerance:
        raise ArtifactVerificationError(
            f"PyTorch/ONNX maximum logit difference {max_difference:.8g} exceeds "
            f"tolerance {tolerance:.1e}."
        )
    torch_predictions = np.argmax(torch_logits, axis=-1)
    onnx_predictions = np.argmax(onnx_logits, axis=-1)
    if not np.array_equal(torch_predictions, onnx_predictions):
        raise ArtifactVerificationError("PyTorch and ONNX top-1 predictions differ.")

    lines.extend([
        f"PASS finite logits and normalized softmax probabilities; sample={sample_source}",
        f"PASS PyTorch/ONNX parity; max_abs_logit_difference={max_difference:.8g}; "
        f"tolerance={tolerance:.1e}; top1_predictions_match={len(sample)}",
    ])
    for path in (checkpoint_path, onnx_path, vocabulary_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"SHA256 {path}: {digest}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--sample-data", type=Path, default=DEFAULT_SAMPLE_DATA)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--tolerance", type=float, default=PARITY_TOLERANCE)
    args = parser.parse_args()

    try:
        lines = verify_word_model_artifacts(
            checkpoint_path=args.checkpoint,
            onnx_path=args.onnx,
            vocabulary_path=args.vocabulary,
            sample_data_path=args.sample_data,
            metadata_path=args.metadata,
            tolerance=args.tolerance,
        )
    except (ArtifactVerificationError, OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    for line in lines:
        print(line)
    print("PASS all word-model artifact checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
