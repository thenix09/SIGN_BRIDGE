"""Unit tests for the production ONNXWordRecognizer runtime adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from backend.app.core.exceptions import InvalidLandmarkError, ModelNotAvailableError
from backend.app.domain.sequence import LandmarkFrame, LandmarkWindow
from backend.app.recognition.base import WordRecognizer
from backend.app.recognition.onnx_word_model import ONNXWordRecognizer

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


def _build_dummy_window(num_frames: int = 15, ox: float = 0.5, oy: float = 0.5) -> LandmarkWindow:
    """Helper creating a valid LandmarkWindow with pose and both hands."""
    frames: list[LandmarkFrame] = []
    sl_x, sr_x = ox - 0.1, ox + 0.1
    s_y = oy

    for _ in range(num_frames):
        pose = (
            (ox, oy - 0.15),  # 0: nose
            (ox - 0.03, oy - 0.08),  # 1: mouth L
            (ox + 0.03, oy - 0.08),  # 2: mouth R
            (sl_x, s_y),  # 3: shoulder L
            (sr_x, s_y),  # 4: shoulder R
            (sl_x - 0.05, s_y + 0.15),  # 5: elbow L
            (sr_x + 0.05, s_y + 0.15),  # 6: elbow R
            (sl_x - 0.08, s_y + 0.30),  # 7: wrist L
            (sr_x + 0.08, s_y + 0.30),  # 8: wrist R
        )
        lh = tuple((sl_x - 0.08 + float(i) * 0.003, s_y + 0.30) for i in range(21))
        rh = tuple((sr_x + 0.08 + float(i) * 0.003, s_y + 0.30) for i in range(21))
        frames.append(LandmarkFrame(pose=pose, left_hand=lh, right_hand=rh))

    return LandmarkWindow(frames)


def _create_mock_onnx_model(
    path: Path,
    in_shape: list[int | str | None],
    out_shape: list[int | str | None] | None = None,
) -> Path:
    """Helper creating a small valid ONNX model file with specified node shapes.

    If *out_shape* is omitted it falls back to *in_shape* (legacy behaviour).
    The model uses an Identity op so in_shape and out_shape must match for a
    runnable session; for shape-validation-only tests this is fine.
    """
    if out_shape is None:
        out_shape = in_shape
    input_node = helper.make_tensor_value_info("landmarks", TensorProto.FLOAT, in_shape)
    output_node = helper.make_tensor_value_info("logits", TensorProto.FLOAT, out_shape)

    node = helper.make_node("Identity", inputs=["landmarks"], outputs=["logits"])
    graph = helper.make_graph([node], "test_graph", [input_node], [output_node])
    model = helper.make_model(
        graph,
        producer_name="test_producer",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    onnx.save(model, str(path))
    return path


def _create_dynamic_logits_model(path: Path) -> Path:
    """Create a dynamic-batch model that returns deterministic descending logits."""
    weights = helper.make_tensor("weights", TensorProto.FLOAT, [1, 30], [0.0] * 30)
    bias = helper.make_tensor(
        "bias",
        TensorProto.FLOAT,
        [30],
        [float(30 - index) for index in range(30)],
    )
    nodes = [
        helper.make_node(
            "ReduceMean",
            inputs=["landmarks"],
            outputs=["reduced"],
            axes=[1, 2],
            keepdims=1,
        ),
        helper.make_node("Flatten", inputs=["reduced"], outputs=["flat"], axis=1),
        helper.make_node("Gemm", inputs=["flat", "weights", "bias"], outputs=["logits"]),
    ]
    input_node = helper.make_tensor_value_info(
        "landmarks", TensorProto.FLOAT, ["batch", 32, 104]
    )
    output_node = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 30])
    graph = helper.make_graph(
        nodes,
        "deterministic_logits_graph",
        [input_node],
        [output_node],
        initializer=[weights, bias],
    )
    model = helper.make_model(
        graph,
        producer_name="test_producer",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _make_vocab_file(path: Path, vocabulary: list, label_to_id: object = None) -> Path:
    """Write a vocab JSON file with the given vocabulary and optional label_to_id."""
    payload: dict = {"vocabulary": vocabulary}
    if label_to_id is not None:
        payload["label_to_id"] = label_to_id
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_test_recognizer(tmp_path: Path) -> ONNXWordRecognizer:
    """Create a recognizer backed by temporary deterministic artifacts."""
    vocabulary = [f"word_{index}" for index in range(30)]
    vocab_path = _make_vocab_file(tmp_path / "vocab.json", vocabulary)
    model_path = _create_dynamic_logits_model(tmp_path / "model.onnx")
    return ONNXWordRecognizer(model_path=model_path, vocab_path=vocab_path)


def test_onnx_word_recognizer_polymorphism() -> None:
    """Verify ONNXWordRecognizer satisfies WordRecognizer ABC."""
    recognizer = ONNXWordRecognizer()
    assert isinstance(recognizer, WordRecognizer)
    assert recognizer.language == "ASL"
    assert isinstance(recognizer.vocabulary, tuple)
    assert len(recognizer.vocabulary) == 30


def test_safe_loading_and_idempotence(tmp_path: Path) -> None:
    """Verify load() loads successfully and repeated calls do not re-initialize."""
    recognizer = _make_test_recognizer(tmp_path)
    assert not recognizer.is_loaded

    recognizer.load()
    assert recognizer.is_loaded

    session_before = recognizer._session
    recognizer.load()
    assert recognizer.is_loaded
    assert recognizer._session is session_before


def test_missing_files_error_handling(tmp_path: Path) -> None:
    """Verify ModelNotAvailableError is raised when model or vocab files are missing."""
    missing_model = tmp_path / "missing.onnx"
    missing_vocab = tmp_path / "missing.json"

    rec1 = ONNXWordRecognizer(model_path=missing_model)
    with pytest.raises(ModelNotAvailableError, match="model file not found"):
        rec1.load()
    # Verify rollback on missing model file
    assert not rec1.is_loaded
    assert rec1._session is None
    assert rec1._vocabulary is None

    rec2 = ONNXWordRecognizer(vocab_path=missing_vocab)
    with pytest.raises(ModelNotAvailableError, match="vocabulary file not found"):
        rec2.load()
    assert not rec2.is_loaded
    assert rec2._vocabulary is None


def test_vocabulary_metadata_validation(tmp_path: Path) -> None:
    """Verify validation of vocabulary length, uniqueness, and label_to_id mapping."""
    vocab_file = tmp_path / "vocab.json"

    # Case 1: Wrong number of classes (< 30)
    vocab_file.write_text(json.dumps({"vocabulary": ["hello", "bye"]}), encoding="utf-8")
    rec = ONNXWordRecognizer(vocab_path=vocab_file)
    with pytest.raises(ModelNotAvailableError, match="exactly 30 labels"):
        rec.load()
    assert not rec.is_loaded
    assert rec._vocabulary is None

    # Case 2: Duplicate labels
    words_with_dup = [f"word_{i}" for i in range(29)] + ["word_0"]
    vocab_file.write_text(json.dumps({"vocabulary": words_with_dup}), encoding="utf-8")
    rec = ONNXWordRecognizer(vocab_path=vocab_file)
    with pytest.raises(ModelNotAvailableError, match="duplicate labels"):
        rec.load()
    assert not rec.is_loaded
    assert rec._vocabulary is None

    # Case 3: Inconsistent label_to_id mapping
    valid_words = [f"word_{i}" for i in range(30)]
    bad_map = {word: i for i, word in enumerate(reversed(valid_words))}
    vocab_file.write_text(json.dumps({"vocabulary": valid_words, "label_to_id": bad_map}), encoding="utf-8")
    rec = ONNXWordRecognizer(vocab_path=vocab_file)
    with pytest.raises(ModelNotAvailableError, match="label_to_id mapping disagrees"):
        rec.load()
    assert not rec.is_loaded
    assert rec._vocabulary is None


def test_invalid_onnx_node_dimensions_and_corrupt_file(tmp_path: Path) -> None:
    """Verify load fails with rollback when ONNX node shapes are invalid or file is corrupt."""
    vocab_file = tmp_path / "vocab.json"
    valid_words = [f"word_{i}" for i in range(30)]
    vocab_file.write_text(
        json.dumps({
            "vocabulary": valid_words,
            "label_to_id": {w: i for i, w in enumerate(valid_words)},
        }),
        encoding="utf-8",
    )

    # Case 1: Wrong input shape (e.g. [batch, 20, 50] instead of [batch, 32, 104])
    bad_input_model = tmp_path / "bad_input.onnx"
    _create_mock_onnx_model(bad_input_model, in_shape=["batch", 20, 50], out_shape=["batch", 20, 50])
    rec1 = ONNXWordRecognizer(model_path=bad_input_model, vocab_path=vocab_file)
    with pytest.raises(ModelNotAvailableError, match="input node shape"):
        rec1.load()
    assert not rec1.is_loaded
    assert rec1._session is None
    assert rec1._vocabulary is None

    # Case 2: Wrong output shape (e.g. [batch, 32, 104] instead of [batch, 30])
    bad_output_model = tmp_path / "bad_output.onnx"
    _create_mock_onnx_model(bad_output_model, in_shape=["batch", 32, 104], out_shape=["batch", 32, 104])
    rec2 = ONNXWordRecognizer(model_path=bad_output_model, vocab_path=vocab_file)
    with pytest.raises(ModelNotAvailableError, match="output node shape"):
        rec2.load()
    assert not rec2.is_loaded
    assert rec2._session is None
    assert rec2._vocabulary is None

    # Case 3: Corrupt binary
    corrupt_model = tmp_path / "corrupt.onnx"
    corrupt_model.write_bytes(b"corrupt binary bytes not a real onnx model")
    rec3 = ONNXWordRecognizer(model_path=corrupt_model, vocab_path=vocab_file)
    with pytest.raises(ModelNotAvailableError, match="Failed to initialize ONNX runtime"):
        rec3.load()
    assert not rec3.is_loaded
    assert rec3._session is None
    assert rec3._vocabulary is None


def test_top_k_boundary_handling(tmp_path: Path) -> None:
    """Verify top_k boundary cases: zero, negative, exact, and oversized."""
    recognizer = _make_test_recognizer(tmp_path)
    window = _build_dummy_window(num_frames=20)

    # Zero and negative top_k return empty list
    assert recognizer.recognize(window, top_k=0) == []
    assert recognizer.recognize(window, top_k=-1) == []
    assert recognizer.recognize(window, top_k=-10) == []

    # top_k = 1
    res_1 = recognizer.recognize(window, top_k=1)
    assert len(res_1) == 1

    # top_k = 30
    res_30 = recognizer.recognize(window, top_k=30)
    assert len(res_30) == 30

    # top_k > 30 (clamped to 30)
    res_oversized = recognizer.recognize(window, top_k=100)
    assert len(res_oversized) == 30
    assert [g.label for g in res_oversized] == [g.label for g in res_30]
    assert [gesture.label for gesture in res_30] == list(recognizer.vocabulary)
    confidences = [gesture.confidence for gesture in res_30]
    assert confidences == sorted(confidences, reverse=True)
    assert math.isclose(sum(confidences), 1.0, rel_tol=1e-6)


def test_onnx_output_validation_nan_and_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that malformed ONNX runtime outputs (wrong shape, NaN, Inf) are safely rejected."""
    recognizer = _make_test_recognizer(tmp_path)
    recognizer.load()
    assert recognizer._session is not None
    window = _build_dummy_window(num_frames=15)

    # Mock session returning NaN logits
    nan_logits = np.full((1, 30), np.nan, dtype=np.float32)
    monkeypatch.setattr(recognizer._session, "run", lambda *args, **kwargs: [nan_logits])
    with pytest.raises(ModelNotAvailableError, match="Non-finite values"):
        recognizer.recognize(window)

    # Mock session returning Inf logits
    inf_logits = np.full((1, 30), np.inf, dtype=np.float32)
    monkeypatch.setattr(recognizer._session, "run", lambda *args, **kwargs: [inf_logits])
    with pytest.raises(ModelNotAvailableError, match="Non-finite values"):
        recognizer.recognize(window)

    # Mock session returning mismatched output class shape (1, 10)
    wrong_shape_logits = np.zeros((1, 10), dtype=np.float32)
    monkeypatch.setattr(recognizer._session, "run", lambda *args, **kwargs: [wrong_shape_logits])
    with pytest.raises(ModelNotAvailableError, match="Unexpected ONNX output shape"):
        recognizer.recognize(window)


def test_recognize_parity_with_direct_onnx() -> None:
    """Verify recognizer output matches raw direct ONNX session inference."""
    model_path = MODELS_DIR / "word_model.onnx"
    if not model_path.is_file():
        pytest.skip(f"Production ONNX model not found at {model_path}")

    recognizer = ONNXWordRecognizer(model_path=model_path)
    recognizer.load()
    assert recognizer._session is not None

    window = _build_dummy_window(num_frames=18)
    tensor = window.to_tensor()
    raw_input = np.expand_dims(tensor, axis=0).astype(np.float32)

    # Direct ONNX run
    direct_logits = recognizer._session.run(
        [recognizer._output_name],
        {recognizer._input_name: raw_input},
    )[0][0]
    exp = np.exp(direct_logits - np.max(direct_logits))
    direct_probs = exp / np.sum(exp)
    top_idx = int(np.argmax(direct_probs))
    expected_top_label = recognizer.vocabulary[top_idx]
    expected_top_conf = float(direct_probs[top_idx])

    # Recognizer run
    results = recognizer.recognize(window, top_k=1)
    assert len(results) == 1
    assert results[0].label == expected_top_label
    assert math.isclose(results[0].confidence, expected_top_conf, rel_tol=1e-5)


def test_invalid_input_error_handling(tmp_path: Path) -> None:
    """Verify InvalidLandmarkError is raised when passed non-LandmarkWindow objects."""
    recognizer = _make_test_recognizer(tmp_path)

    with pytest.raises(InvalidLandmarkError, match="Expected LandmarkWindow"):
        recognizer.recognize(window=None)  # type: ignore[arg-type]

    with pytest.raises(InvalidLandmarkError, match="Expected LandmarkWindow"):
        recognizer.recognize(window="invalid")  # type: ignore[arg-type]


def test_vocab_non_string_label(tmp_path: Path) -> None:
    """Vocabulary containing a non-string element (integer) must be rejected."""
    vocab = [f"word_{i}" for i in range(29)] + [42]  # integer at position 29
    vocab_file = _make_vocab_file(tmp_path / "vocab.json", vocab)
    rec = ONNXWordRecognizer(vocab_path=vocab_file)
    with pytest.raises(ModelNotAvailableError, match="non-empty strings"):
        rec.load()
    assert not rec.is_loaded
    assert rec._vocabulary is None


def test_vocab_empty_string_label(tmp_path: Path) -> None:
    """Vocabulary containing an empty string must be rejected."""
    vocab = [f"word_{i}" for i in range(29)] + [""]  # empty string at position 29
    vocab_file = _make_vocab_file(tmp_path / "vocab.json", vocab)
    rec = ONNXWordRecognizer(vocab_path=vocab_file)
    with pytest.raises(ModelNotAvailableError, match="non-empty strings"):
        rec.load()
    assert not rec.is_loaded
    assert rec._vocabulary is None


def test_label_to_id_not_dict(tmp_path: Path) -> None:
    """label_to_id that is not a dict (e.g. a list) must be rejected."""
    vocab = [f"word_{i}" for i in range(30)]
    # Pass a list instead of a dict for label_to_id
    vocab_file = _make_vocab_file(tmp_path / "vocab.json", vocab, label_to_id=[0, 1, 2])
    rec = ONNXWordRecognizer(vocab_path=vocab_file)
    with pytest.raises(ModelNotAvailableError, match="must be a dictionary"):
        rec.load()
    assert not rec.is_loaded
    assert rec._vocabulary is None


def test_onnx_dynamic_batch_dimension(tmp_path: Path) -> None:
    """The loader accepts a symbolic batch dimension on a temporary ONNX model."""
    recognizer = _make_test_recognizer(tmp_path)
    recognizer.load()
    assert recognizer.is_loaded
    assert recognizer._session is not None
    in_shape = recognizer._session.get_inputs()[0].shape
    out_shape = recognizer._session.get_outputs()[0].shape
    assert in_shape == ["batch", 32, 104]
    assert out_shape == ["batch", 30]


def test_onnx_fixed_batch_dimension(tmp_path: Path) -> None:
    """ONNX model with a fixed batch dimension (1, 32, 104) -> (1, 30) must load successfully.

    Uses a minimal constant-output model so ORT can correctly infer output shape.
    """
    valid_words = [f"word_{i}" for i in range(30)]
    vocab_file = _make_vocab_file(
        tmp_path / "vocab.json",
        valid_words,
        label_to_id={w: i for i, w in enumerate(valid_words)},
    )

    # Build a minimal ONNX model: Flatten input (1,32,104) -> (1,3328) then
    # MatMul with a constant weight (3328,30) -> (1,30).
    weight = np.zeros((32 * 104, 30), dtype=np.float32)
    weight_tensor = helper.make_tensor(
        "W", TensorProto.FLOAT, [32 * 104, 30], weight.flatten().tolist()
    )

    # Flatten input to (1, 32*104)
    flatten_node = helper.make_node("Flatten", inputs=["landmarks"], outputs=["flat"], axis=1)
    # MatMul flat (1, 3328) @ W (3328, 30) -> (1, 30)
    matmul_node = helper.make_node("MatMul", inputs=["flat", "W"], outputs=["logits"])

    input_vi = helper.make_tensor_value_info("landmarks", TensorProto.FLOAT, [1, 32, 104])
    output_vi = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 30])

    graph = helper.make_graph(
        [flatten_node, matmul_node],
        "fixed_batch_graph",
        [input_vi],
        [output_vi],
        initializer=[weight_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name="test",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    model_path = tmp_path / "fixed_batch.onnx"
    onnx.save(model, str(model_path))

    rec = ONNXWordRecognizer(model_path=model_path, vocab_path=vocab_file)
    rec.load()
    assert rec.is_loaded
    assert rec._vocabulary is not None
    assert len(rec._vocabulary) == 30
    assert rec._session is not None
    in_shape = rec._session.get_inputs()[0].shape
    out_shape = rec._session.get_outputs()[0].shape
    assert in_shape == [1, 32, 104]
    assert out_shape == [1, 30]


def test_onnx_invalid_batch_dimensions(tmp_path: Path) -> None:
    """ONNX model with invalid rank or dimensions (e.g. 2D or 4D) must be rejected."""
    valid_words = [f"word_{i}" for i in range(30)]
    vocab_file = _make_vocab_file(
        tmp_path / "vocab.json",
        valid_words,
        label_to_id={w: i for i, w in enumerate(valid_words)},
    )

    # Case 1: 2D input (missing batch dimension, shape [32, 104])
    model_2d = tmp_path / "model_2d.onnx"
    _create_mock_onnx_model(model_2d, in_shape=[32, 104], out_shape=[32, 104])
    rec_2d = ONNXWordRecognizer(model_path=model_2d, vocab_path=vocab_file)
    with pytest.raises(ModelNotAvailableError, match="input node shape"):
        rec_2d.load()
    assert not rec_2d.is_loaded

    # Case 2: 4D input (extra dimension, shape [1, 1, 32, 104])
    model_4d = tmp_path / "model_4d.onnx"
    _create_mock_onnx_model(model_4d, in_shape=[1, 1, 32, 104], out_shape=[1, 1, 32, 104])
    rec_4d = ONNXWordRecognizer(model_path=model_4d, vocab_path=vocab_file)
    with pytest.raises(ModelNotAvailableError, match="input node shape"):
        rec_4d.load()
    assert not rec_4d.is_loaded
