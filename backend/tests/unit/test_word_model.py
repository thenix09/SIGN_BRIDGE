"""Unit tests for WordSignModel architecture, vocabulary mapping, and checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from backend.app.recognition.word_model import WordSignModel

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


def test_word_sign_model_architecture_and_output_shape() -> None:
    """Verify WordSignModel architecture, parameter count, and output dimensions."""
    model = WordSignModel(
        in_features=104,
        num_classes=30,
        cnn_channels=(128, 128),
        gru_hidden_size=128,
        num_gru_layers=2,
        dropout=0.3,
    )

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert model.in_features == 104
    assert model.num_classes == 30
    assert model.gru.hidden_size == 128
    assert model.gru.bidirectional
    assert model.gru.num_layers == 2
    assert trainable_parameters == 653_982
    assert trainable_parameters < 2_000_000

    # Test forward pass with batch size 2
    dummy_input = torch.zeros(2, 32, 104)
    logits = model(dummy_input)
    assert logits.shape == (2, 30)
    assert torch.isfinite(logits).all()


def test_deterministic_vocabulary_mapping() -> None:
    """Verify word_vocab.json mappings against the canonical vocabulary."""
    raw_vocab_path = DATA_DIR / "vocabulary.json"
    with open(raw_vocab_path, encoding="utf-8") as f:
        raw_vocab = json.load(f)
    words = [w["label"] for w in raw_vocab["words"]]
    assert len(words) == 30
    assert len(set(words)) == 30

    sorted_words = sorted(words)
    expected_label_to_id = {word: idx for idx, word in enumerate(sorted_words)}
    expected_id_to_label = {str(idx): word for idx, word in enumerate(sorted_words)}

    # Tracked word_vocab.json is mandatory
    vocab_file = MODELS_DIR / "word_vocab.json"
    assert vocab_file.is_file(), f"Tracked word vocabulary metadata missing at {vocab_file}"
    with open(vocab_file, encoding="utf-8") as f:
        saved_vocab = json.load(f)
    assert saved_vocab["vocabulary"] == sorted_words
    assert saved_vocab["label_to_id"] == expected_label_to_id
    assert saved_vocab["id_to_label"] == expected_id_to_label

    # Generated meta.json remains optional (only present after offline preprocessing)
    meta_file = DATA_DIR / "processed" / "meta.json"
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["vocabulary"] == sorted_words
        assert meta["label_to_id"] == expected_label_to_id
        assert meta["id_to_label"] == expected_id_to_label


def test_synthetic_checkpoint_contract_and_inference(tmp_path: Path) -> None:
    """Verify checkpoint save/load contract and inference deterministically using tmp_path."""
    model = WordSignModel(
        in_features=104,
        num_classes=30,
        cnn_channels=(128, 128),
        gru_hidden_size=128,
        num_gru_layers=2,
        dropout=0.3,
    )
    model.eval()

    vocabulary = [f"word_{i}" for i in range(30)]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ckpt_path = tmp_path / "synthetic_word_model.pt"
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "vocabulary": vocabulary,
        "num_classes": 30,
        "total_params": trainable_params,
    }
    torch.save(checkpoint_data, ckpt_path)

    loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert set(loaded_ckpt) == {
        "model_state_dict",
        "vocabulary",
        "num_classes",
        "total_params",
    }
    assert loaded_ckpt["vocabulary"] == vocabulary
    assert loaded_ckpt["num_classes"] == 30
    assert loaded_ckpt["total_params"] == trainable_params == 653_982

    loaded_model = WordSignModel(
        in_features=104,
        num_classes=30,
        cnn_channels=(128, 128),
        gru_hidden_size=128,
        num_gru_layers=2,
        dropout=0.3,
    )
    loaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
    loaded_model.eval()

    loaded_parameters = sum(
        parameter.numel()
        for parameter in loaded_model.parameters()
        if parameter.requires_grad
    )
    assert loaded_model.in_features == 104
    assert loaded_model.num_classes == 30
    assert loaded_model.gru.hidden_size == 128
    assert loaded_model.gru.bidirectional
    assert loaded_model.gru.num_layers == 2
    assert loaded_parameters == loaded_ckpt["total_params"]

    sample_input = torch.randn(3, 32, 104)
    with torch.inference_mode():
        output = loaded_model(sample_input)
    assert output.shape == (3, 30)
    assert torch.isfinite(output).all()


def test_synthetic_onnx_export_contract_and_dynamic_batching(tmp_path: Path) -> None:
    """Verify ONNX export, graph validity, and dynamic batching deterministically using tmp_path."""
    model = WordSignModel(
        in_features=104,
        num_classes=30,
        cnn_channels=(128, 128),
        gru_hidden_size=128,
        num_gru_layers=2,
        dropout=0.3,
    )
    model.eval()

    onnx_path = tmp_path / "synthetic_word_model.onnx"
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

    exported_model = onnx.load(onnx_path)
    onnx.checker.check_model(exported_model, full_check=True)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    assert len(inputs) == 1
    assert inputs[0].name == "landmarks"
    assert inputs[0].shape[1] == 32
    assert inputs[0].shape[2] == 104

    assert len(outputs) == 1
    assert outputs[0].name == "logits"
    assert outputs[0].shape[1] == 30

    for b in [1, 4]:
        dummy = torch.randn(b, 32, 104).numpy().astype("float32")
        result = session.run([outputs[0].name], {inputs[0].name: dummy})
        assert result[0].shape == (b, 30)
        assert np.isfinite(result[0]).all()


def test_checkpoint_contract_and_inference() -> None:
    """Verify that the production PyTorch checkpoint matches the architecture and executes cleanly."""
    checkpoint_path = MODELS_DIR / "word_model.pt"
    if not checkpoint_path.is_file():
        pytest.skip(f"Production checkpoint file not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert "model_state_dict" in checkpoint
    assert "vocabulary" in checkpoint
    assert len(checkpoint["vocabulary"]) == 30
    with (MODELS_DIR / "word_vocab.json").open(encoding="utf-8") as f:
        saved_vocab = json.load(f)
    assert checkpoint["vocabulary"] == saved_vocab["vocabulary"]
    assert checkpoint["num_classes"] == 30
    assert checkpoint["total_params"] == 653_982

    model = WordSignModel(num_classes=30)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_input = torch.randn(3, 32, 104)
    with torch.no_grad():
        output = model(sample_input)
    assert output.shape == (3, 30)
    assert torch.isfinite(output).all()


def test_onnx_model_contract_and_dynamic_batching() -> None:
    """Verify exported production ONNX model schema and dynamic batching."""
    onnx_path = MODELS_DIR / "word_model.onnx"
    if not onnx_path.is_file():
        pytest.skip(f"Production ONNX model not found at {onnx_path}")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model, full_check=True)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    assert len(inputs) == 1
    assert inputs[0].name == "landmarks"
    # Dynamic batch dimension, 32 frames, 104 features
    assert inputs[0].shape[1] == 32
    assert inputs[0].shape[2] == 104

    assert len(outputs) == 1
    assert outputs[0].name == "logits"
    assert outputs[0].shape[1] == 30

    # Test dynamic batching across batch sizes 1 and 4
    for b in [1, 4]:
        dummy = torch.randn(b, 32, 104).numpy().astype("float32")
        result = session.run([outputs[0].name], {inputs[0].name: dummy})
        assert result[0].shape == (b, 30)
        assert np.isfinite(result[0]).all()
