"""Tests for alphabet model artifact verification with temporary models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from backend.scripts import verify_alphabet_model_artifacts as verifier

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@pytest.fixture
def alphabet_artifacts(tmp_path: Path) -> dict[str, Any]:
    """Create a small fitted 300-tree model and matching temporary metadata."""
    model_path = tmp_path / "asl_model.joblib"
    metadata_path = tmp_path / "letters_meta.json"
    class_indices = np.tile(np.arange(26), 4)
    features = np.zeros((len(class_indices), 63), dtype=np.float32)
    features[np.arange(len(class_indices)), class_indices] = 1.0
    labels = np.asarray(LABELS)[class_indices]
    model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=2)
    model.fit(features, labels)
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps(
            {
                "labels": LABELS,
                "label_to_id": {label: index for index, label in enumerate(LABELS)},
                "id_to_label": {str(index): label for index, label in enumerate(LABELS)},
                "canonical_handedness": "Right",
                "feature_count": 63,
            }
        ),
        encoding="utf-8",
    )
    return {
        "model": model_path,
        "metadata": metadata_path,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }


def _verify(artifacts: dict[str, Any], **overrides: Any) -> list[str]:
    options = {
        "model_path": artifacts["model"],
        "metadata_path": artifacts["metadata"],
        "expected_sha256": artifacts["sha256"],
    }
    options.update(overrides)
    return verifier.verify_alphabet_model_artifacts(**options)


def test_verifies_valid_temporary_model(alphabet_artifacts: dict[str, Any]) -> None:
    """A valid temporary Random Forest passes every artifact check."""
    results = _verify(alphabet_artifacts)

    assert any("SHA-256 matches" in result for result in results)
    assert any("feature count=63; class count=26; classes=A-Z" in result for result in results)
    assert any("26 finite normalized probabilities" in result for result in results)
    assert any("prediction is valid A-Z label" in result for result in results)


def test_rejects_model_class_order_mismatch(
    alphabet_artifacts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model whose class order differs from metadata is rejected."""
    model = joblib.load(alphabet_artifacts["model"])
    model.classes_ = model.classes_[::-1]
    monkeypatch.setattr(verifier.joblib, "load", lambda _path: model)

    with pytest.raises(verifier.ArtifactVerificationError, match="classes must match"):
        _verify(alphabet_artifacts)


def test_rejects_wrong_feature_count(tmp_path: Path) -> None:
    """An alphabet classifier trained on 62 features fails the 63-feature contract."""
    model_path = tmp_path / "wrong_features.joblib"
    class_indices = np.tile(np.arange(26), 4)
    features = np.zeros((len(class_indices), 62), dtype=np.float32)
    features[np.arange(len(class_indices)), class_indices] = 1.0
    labels = np.asarray(LABELS)[class_indices]
    model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=2)
    model.fit(features, labels)
    joblib.dump(model, model_path)
    metadata_path = tmp_path / "letters_meta.json"
    metadata_path.write_text(
        json.dumps(
            {
                "labels": LABELS,
                "label_to_id": {label: index for index, label in enumerate(LABELS)},
                "id_to_label": {str(index): label for index, label in enumerate(LABELS)},
                "canonical_handedness": "Right",
                "feature_count": 63,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(verifier.ArtifactVerificationError, match="Expected 63 input features"):
        verifier.verify_alphabet_model_artifacts(
            model_path,
            metadata_path,
            expected_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        )


def test_rejects_missing_model(alphabet_artifacts: dict[str, Any], tmp_path: Path) -> None:
    """A missing model fails explicitly without attempting to load it."""
    with pytest.raises(verifier.ArtifactVerificationError, match="model file not found"):
        verifier.verify_alphabet_model_artifacts(
            tmp_path / "missing.joblib",
            alphabet_artifacts["metadata"],
            expected_sha256=alphabet_artifacts["sha256"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("labels", list(reversed(LABELS)), "labels must be ordered"),
        ("canonical_handedness", "Left", "canonical_handedness"),
        ("feature_count", 62, "feature_count"),
    ],
)
def test_rejects_invalid_metadata(
    alphabet_artifacts: dict[str, Any], field: str, value: Any, message: str
) -> None:
    """Invalid class order, handedness, or feature metadata is rejected."""
    metadata = json.loads(alphabet_artifacts["metadata"].read_text(encoding="utf-8"))
    metadata[field] = value
    alphabet_artifacts["metadata"].write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(verifier.ArtifactVerificationError, match=message):
        _verify(alphabet_artifacts)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.full((1, 26), np.nan), "non-finite"),
        (np.full((1, 26), 0.1), "not normalized"),
    ],
)
def test_rejects_invalid_probability_output(
    alphabet_artifacts: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    probabilities: np.ndarray,
    message: str,
) -> None:
    """Non-finite or non-normalized probabilities fail validation."""

    class InvalidProbabilityForest(RandomForestClassifier):
        def __init__(self) -> None:
            super().__init__(n_estimators=300)
            self.estimators_ = [object()] * 300
            self.n_features_in_ = 63
            self.classes_ = np.asarray(LABELS)

        def predict_proba(self, _features: np.ndarray) -> np.ndarray:
            return probabilities

        def predict(self, _features: np.ndarray) -> np.ndarray:
            return np.asarray(["A"])

    monkeypatch.setattr(verifier.joblib, "load", lambda _path: InvalidProbabilityForest())

    with pytest.raises(verifier.ArtifactVerificationError, match=message):
        _verify(alphabet_artifacts)


def test_rejects_checksum_mismatch(alphabet_artifacts: dict[str, Any]) -> None:
    """A model with any checksum other than the required digest is rejected."""
    with pytest.raises(verifier.ArtifactVerificationError, match="SHA-256 mismatch"):
        _verify(alphabet_artifacts, expected_sha256="0" * 64)
