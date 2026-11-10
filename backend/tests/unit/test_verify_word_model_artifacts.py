"""Tests for word-model artifact verification using temporary synthetic artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from backend.app.recognition.word_model import WordSignModel
from backend.scripts.verify_word_model_artifacts import (
    ArtifactVerificationError,
    verify_word_model_artifacts,
)


@pytest.fixture(scope="module")
def synthetic_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Create a checkpoint, ONNX export, vocabulary, and input sample in a temp directory."""
    directory = tmp_path_factory.mktemp("word-model-artifacts")
    vocabulary = [f"word_{index:02d}" for index in range(30)]
    label_to_id = {label: index for index, label in enumerate(vocabulary)}
    id_to_label = {str(index): label for index, label in enumerate(vocabulary)}
    vocabulary_path = directory / "word_vocab.json"
    vocabulary_path.write_text(
        json.dumps(
            {
                "vocabulary": vocabulary,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
            }
        ),
        encoding="utf-8",
    )

    torch.manual_seed(17)
    model = WordSignModel().eval()
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    checkpoint_path = directory / "word_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "num_classes": len(vocabulary),
            "total_params": parameter_count,
        },
        checkpoint_path,
    )

    onnx_path = directory / "word_model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (torch.zeros(2, 32, 104, dtype=torch.float32),),
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["landmarks"],
            output_names=["logits"],
            dynamic_axes={
                "landmarks": {0: "batch"},
                "logits": {0: "batch"},
            },
            dynamo=False,
        )
    onnx_model = onnx.load(onnx_path)
    onnx.helper.set_model_props(
        onnx_model,
        {
            "vocabulary": json.dumps(vocabulary, separators=(",", ":")),
            "label_to_id": json.dumps(label_to_id, separators=(",", ":")),
        },
    )
    onnx.save(onnx_model, onnx_path)

    sample_data_path = directory / "word_dataset.npz"
    np.savez(
        sample_data_path,
        X=np.zeros((2, 32, 104), dtype=np.float32),
        split=np.array(["train", "test"]),
    )
    return {
        "checkpoint": checkpoint_path,
        "onnx": onnx_path,
        "vocabulary": vocabulary_path,
        "sample_data": sample_data_path,
    }


def test_verifies_synthetic_artifacts(synthetic_artifacts: dict[str, Path]) -> None:
    """Valid temporary model artifacts pass loading, contract, and parity checks."""
    lines = verify_word_model_artifacts(
        checkpoint_path=synthetic_artifacts["checkpoint"],
        onnx_path=synthetic_artifacts["onnx"],
        vocabulary_path=synthetic_artifacts["vocabulary"],
        sample_data_path=synthetic_artifacts["sample_data"],
        metadata_path=synthetic_artifacts["sample_data"].with_name("missing-meta.json"),
    )

    assert any(line.startswith("PASS checkpoint loaded") for line in lines)
    assert any("vocabulary ordering matches" in line for line in lines)
    assert any("PASS ONNX checker" in line for line in lines)
    assert any("processed data:" in line for line in lines)
    assert any("PASS PyTorch/ONNX parity" in line for line in lines)
    assert sum(line.startswith("SHA256 ") for line in lines) == 3


def test_rejects_vocabulary_order_mismatch(
    synthetic_artifacts: dict[str, Path], tmp_path: Path
) -> None:
    """An otherwise valid artifact set fails when the vocabulary order changes."""
    source = json.loads(synthetic_artifacts["vocabulary"].read_text(encoding="utf-8"))
    source["vocabulary"].reverse()
    source["label_to_id"] = {
        label: index for index, label in enumerate(source["vocabulary"])
    }
    source["id_to_label"] = {
        str(index): label for index, label in enumerate(source["vocabulary"])
    }
    vocabulary_path = tmp_path / "reordered_vocab.json"
    vocabulary_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ArtifactVerificationError, match="ordering"):
        verify_word_model_artifacts(
            checkpoint_path=synthetic_artifacts["checkpoint"],
            onnx_path=synthetic_artifacts["onnx"],
            vocabulary_path=vocabulary_path,
            sample_data_path=synthetic_artifacts["sample_data"],
            metadata_path=tmp_path / "missing-meta.json",
        )


def test_rejects_malformed_checkpoint(
    synthetic_artifacts: dict[str, Path], tmp_path: Path
) -> None:
    """A corrupt checkpoint produces an explicit verification failure."""
    checkpoint_path = tmp_path / "malformed.pt"
    checkpoint_path.write_bytes(b"not a PyTorch checkpoint")

    with pytest.raises(ArtifactVerificationError, match="Could not load PyTorch checkpoint"):
        verify_word_model_artifacts(
            checkpoint_path=checkpoint_path,
            onnx_path=synthetic_artifacts["onnx"],
            vocabulary_path=synthetic_artifacts["vocabulary"],
            sample_data_path=synthetic_artifacts["sample_data"],
            metadata_path=tmp_path / "missing-meta.json",
        )


def test_rejects_malformed_onnx(
    synthetic_artifacts: dict[str, Path], tmp_path: Path
) -> None:
    """A corrupt ONNX protobuf produces an explicit verification failure."""
    onnx_path = tmp_path / "malformed.onnx"
    onnx_path.write_bytes(b"not an ONNX model")

    with pytest.raises(ArtifactVerificationError, match="ONNX validation failed"):
        verify_word_model_artifacts(
            checkpoint_path=synthetic_artifacts["checkpoint"],
            onnx_path=onnx_path,
            vocabulary_path=synthetic_artifacts["vocabulary"],
            sample_data_path=synthetic_artifacts["sample_data"],
            metadata_path=tmp_path / "missing-meta.json",
        )


def test_uses_deterministic_input_when_processed_data_is_absent(
    synthetic_artifacts: dict[str, Path], tmp_path: Path
) -> None:
    """Missing processed tensors use the documented seeded synthetic sample."""
    lines = verify_word_model_artifacts(
        checkpoint_path=synthetic_artifacts["checkpoint"],
        onnx_path=synthetic_artifacts["onnx"],
        vocabulary_path=synthetic_artifacts["vocabulary"],
        sample_data_path=tmp_path / "absent-dataset.npz",
        metadata_path=tmp_path / "missing-meta.json",
    )

    assert any("deterministic synthetic input" in line for line in lines)
    assert any("PASS PyTorch/ONNX parity" in line for line in lines)
