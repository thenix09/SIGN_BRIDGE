"""Export and verify the selected SignBridge word model as ONNX."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.recognition.word_model import WordSignModel  # noqa: E402
from backend.scripts.word_dataset import load_word_dataset  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "backend" / "models" / "word_model.pt"
DEFAULT_VOCABULARY = ROOT / "backend" / "models" / "word_vocab.json"
DEFAULT_DATASET = ROOT / "backend" / "data" / "processed" / "word_dataset.npz"
DEFAULT_META = ROOT / "backend" / "data" / "processed" / "meta.json"
DEFAULT_OUTPUT = ROOT / "backend" / "models" / "word_model.onnx"
INPUT_NAME = "landmarks"
OUTPUT_NAME = "logits"
PARITY_TOLERANCE = 1e-4
MIN_VALIDATION_SAMPLES = 50
PARITY_SEED = 42


def load_selected_model(
    checkpoint_path: Path, vocabulary_path: Path
) -> tuple[WordSignModel, dict[str, int], dict[str, object]]:
    """Load the selected checkpoint and confirm its class ordering."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with vocabulary_path.open(encoding="utf-8") as file:
        vocabulary_data = json.load(file)

    vocabulary = vocabulary_data.get("vocabulary")
    if not isinstance(vocabulary, list) or len(vocabulary) != 30:
        raise ValueError("Vocabulary file must contain the approved 30 labels.")
    label_to_id = {label: index for index, label in enumerate(vocabulary)}
    id_to_label = {str(index): label for index, label in enumerate(vocabulary)}
    if vocabulary_data.get("label_to_id") != label_to_id:
        raise ValueError("Vocabulary label_to_id does not match vocabulary order.")
    if vocabulary_data.get("id_to_label") != id_to_label:
        raise ValueError("Vocabulary id_to_label does not match vocabulary order.")
    if checkpoint.get("vocabulary") != vocabulary:
        raise ValueError("Checkpoint vocabulary does not match word_vocab.json.")
    if checkpoint.get("num_classes") != len(vocabulary):
        raise ValueError("Checkpoint class count does not match word_vocab.json.")
    if "label_to_id" in checkpoint and checkpoint["label_to_id"] != label_to_id:
        raise ValueError("Checkpoint label ordering does not match word_vocab.json.")

    model = WordSignModel(
        in_features=104,
        num_classes=len(vocabulary),
        cnn_channels=(128, 128),
        gru_hidden_size=128,
        num_gru_layers=2,
        dropout=0.3,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != checkpoint.get("total_params"):
        raise ValueError(
            f"Checkpoint declares {checkpoint.get('total_params')} parameters but "
            f"the loaded model has {parameter_count}."
        )
    return model, label_to_id, checkpoint


def validate_onnx_contract(
    model_path: Path,
    vocabulary: list[str],
    label_to_id: dict[str, int],
) -> None:
    """Check ONNX validity and the model's dynamic-batch input/output contract."""
    exported_model = onnx.load(model_path)
    onnx.checker.check_model(exported_model, full_check=True)
    properties = {property_.key: property_.value for property_ in exported_model.metadata_props}
    if properties.get("vocabulary") != json.dumps(vocabulary, separators=(",", ":")):
        raise ValueError("ONNX metadata does not preserve the approved output class order.")
    if properties.get("label_to_id") != json.dumps(label_to_id, separators=(",", ":")):
        raise ValueError("ONNX metadata label mapping does not match word_vocab.json.")
    graph_input = exported_model.graph.input[0]
    graph_output = exported_model.graph.output[0]
    expected_input = (None, 32, 104)
    actual_input = tuple(
        dimension.dim_value if dimension.dim_value > 0 else None
        for dimension in graph_input.type.tensor_type.shape.dim
    )
    actual_output = tuple(
        dimension.dim_value if dimension.dim_value > 0 else None
        for dimension in graph_output.type.tensor_type.shape.dim
    )
    if graph_input.name != INPUT_NAME or actual_input != expected_input:
        raise ValueError(
            f"Unexpected ONNX input contract: {graph_input.name} {actual_input}."
        )
    if graph_output.name != OUTPUT_NAME or actual_output != (None, len(vocabulary)):
        raise ValueError(
            f"Unexpected ONNX output contract: {graph_output.name} {actual_output}."
        )


def export_and_verify(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    vocabulary_path: Path = DEFAULT_VOCABULARY,
    dataset_path: Path = DEFAULT_DATASET,
    meta_path: Path = DEFAULT_META,
    output_path: Path = DEFAULT_OUTPUT,
    validation_samples: int = MIN_VALIDATION_SAMPLES,
) -> dict[str, float | int | list[int]]:
    """Export the checkpoint and require CPU ONNX parity on deterministic val samples."""
    if validation_samples < MIN_VALIDATION_SAMPLES:
        raise ValueError(
            f"At least {MIN_VALIDATION_SAMPLES} validation samples are required."
        )

    model, label_to_id, checkpoint = load_selected_model(checkpoint_path, vocabulary_path)
    dataset = load_word_dataset(dataset_path, meta_path)
    if label_to_id != {label: index for index, label in enumerate(dataset.vocabulary)}:
        raise ValueError("Vocabulary file does not match the approved dataset label mapping.")

    validation_x, _ = dataset.splits["val"]
    if validation_samples > len(validation_x):
        raise ValueError(
            f"Requested {validation_samples} samples, but validation contains {len(validation_x)}."
        )
    sample_indices = np.random.default_rng(PARITY_SEED).choice(
        len(validation_x), size=validation_samples, replace=False
    )
    sample = np.ascontiguousarray(validation_x[sample_indices], dtype=np.float32)
    sample_tensor = torch.from_numpy(sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="signbridge-onnx-", dir=output_path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / output_path.name
        with torch.inference_mode():
            torch.onnx.export(
                model,
                torch.zeros(2, 32, 104, dtype=torch.float32),
                str(temporary_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                dynamic_axes={
                    INPUT_NAME: {0: "batch"},
                    OUTPUT_NAME: {0: "batch"},
                },
                dynamo=False,
            )

        exported_model = onnx.load(temporary_path)
        onnx.helper.set_model_props(
            exported_model,
            {
                "vocabulary": json.dumps(dataset.vocabulary, separators=(",", ":")),
                "label_to_id": json.dumps(label_to_id, separators=(",", ":")),
                "checkpoint_epoch": str(checkpoint["epoch"]),
                "trainable_parameters": str(checkpoint["total_params"]),
            },
        )
        onnx.save(exported_model, temporary_path)
        validate_onnx_contract(temporary_path, dataset.vocabulary, label_to_id)

        session = ort.InferenceSession(
            str(temporary_path), providers=["CPUExecutionProvider"]
        )
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError(f"ONNX Runtime did not use CPU exclusively: {session.get_providers()}")

        with torch.inference_mode():
            pytorch_output = model(sample_tensor).cpu().numpy()
        onnx_output = session.run(
            [OUTPUT_NAME], {INPUT_NAME: sample}
        )[0]
        if pytorch_output.shape != (validation_samples, len(dataset.vocabulary)):
            raise ValueError(f"Unexpected PyTorch output shape: {pytorch_output.shape}.")
        if onnx_output.shape != pytorch_output.shape:
            raise ValueError(
                f"ONNX output shape {onnx_output.shape} does not match PyTorch "
                f"output shape {pytorch_output.shape}."
            )

        max_abs_difference = float(np.max(np.abs(pytorch_output - onnx_output)))
        pytorch_predictions = np.argmax(pytorch_output, axis=1)
        onnx_predictions = np.argmax(onnx_output, axis=1)
        identical_predictions = int(np.count_nonzero(pytorch_predictions == onnx_predictions))
        if max_abs_difference > PARITY_TOLERANCE:
            raise AssertionError(
                f"Maximum absolute output difference {max_abs_difference:.8g} exceeds "
                f"{PARITY_TOLERANCE:.1e}."
            )
        if identical_predictions != validation_samples:
            raise AssertionError(
                f"Top-1 predictions differ on "
                f"{validation_samples - identical_predictions}/{validation_samples} samples."
            )

        dynamic_batch_sizes = sorted({1, min(7, validation_samples), validation_samples})
        for batch_size in dynamic_batch_sizes:
            batch = sample[:batch_size]
            with torch.inference_mode():
                torch_batch_output = model(torch.from_numpy(batch)).cpu().numpy()
            onnx_batch_output = session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0]
            batch_difference = float(np.max(np.abs(torch_batch_output - onnx_batch_output)))
            if batch_difference > PARITY_TOLERANCE:
                raise AssertionError(
                    f"Batch size {batch_size} has maximum absolute output difference "
                    f"{batch_difference:.8g}, exceeding {PARITY_TOLERANCE:.1e}."
                )
            if not np.array_equal(
                np.argmax(torch_batch_output, axis=1), np.argmax(onnx_batch_output, axis=1)
            ):
                raise AssertionError(f"Top-1 predictions differ at batch size {batch_size}.")

        os.replace(temporary_path, output_path)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "trainable_parameters": parameter_count,
        "input_shape": [validation_samples, 32, 104],
        "output_shape": [validation_samples, len(dataset.vocabulary)],
        "validation_samples": validation_samples,
        "max_absolute_difference": max_abs_difference,
        "identical_top1_predictions": identical_predictions,
        "verified_dynamic_batch_sizes": dynamic_batch_sizes,
        "checkpoint_epoch": int(checkpoint["epoch"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-samples", type=int, default=MIN_VALIDATION_SAMPLES)
    args = parser.parse_args()

    results = export_and_verify(
        checkpoint_path=args.checkpoint,
        vocabulary_path=args.vocabulary,
        dataset_path=args.dataset,
        meta_path=args.meta,
        output_path=args.output,
        validation_samples=args.validation_samples,
    )
    print(f"Exported and verified: {args.output}")
    print(f"Checkpoint epoch: {results['checkpoint_epoch']}")
    print(f"Trainable parameters: {results['trainable_parameters']:,}")
    print(f"Input shape: {results['input_shape']} (batch is dynamic)")
    print(f"Output shape: {results['output_shape']} (batch is dynamic)")
    print(f"Validation samples: {results['validation_samples']}")
    print(f"Maximum absolute difference: {results['max_absolute_difference']:.8g}")
    print(
        "Identical top-1 predictions: "
        f"{results['identical_top1_predictions']}/{results['validation_samples']}"
    )
    print(f"Verified dynamic batch sizes: {results['verified_dynamic_batch_sizes']}")


if __name__ == "__main__":
    main()
