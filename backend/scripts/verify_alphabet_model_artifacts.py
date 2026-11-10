"""Verify the provisioned ASL alphabet Random Forest and its metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402

EXPECTED_SHA256 = "d1b0600c19505f85f8c4077f0868fc01ba92414a4e737dd1721f60808c7dfc11"
EXPECTED_ESTIMATORS = 300
FEATURE_COUNT = 63
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_MODEL = ROOT / "backend" / "models" / "asl_model.joblib"
DEFAULT_METADATA = ROOT / "backend" / "models" / "letters_meta.json"


class ArtifactVerificationError(RuntimeError):
    """Raised when an alphabet model artifact fails verification."""


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactVerificationError(f"Metadata file not found: {path}")
    try:
        with path.open(encoding="utf-8") as file:
            metadata = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactVerificationError(f"Could not read metadata {path}: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ArtifactVerificationError("Alphabet metadata must be a JSON object.")
    if metadata.get("labels") != list(LABELS):
        raise ArtifactVerificationError("Metadata labels must be ordered A through Z.")
    expected_mapping = {label: index for index, label in enumerate(LABELS)}
    if metadata.get("label_to_id") != expected_mapping:
        raise ArtifactVerificationError("Metadata label_to_id must map A-Z to indices 0-25.")
    if metadata.get("id_to_label") != {
        str(index): label for index, label in enumerate(LABELS)
    }:
        raise ArtifactVerificationError("Metadata id_to_label must map indices 0-25 to A-Z.")
    if metadata.get("canonical_handedness") != "Right":
        raise ArtifactVerificationError(
            "Metadata canonical_handedness must be exactly 'Right'."
        )
    if metadata.get("feature_count") != FEATURE_COUNT:
        raise ArtifactVerificationError(
            f"Metadata feature_count must be {FEATURE_COUNT}."
        )
    return metadata


def verify_alphabet_model_artifacts(
    model_path: Path = DEFAULT_MODEL,
    metadata_path: Path = DEFAULT_METADATA,
    *,
    expected_sha256: str | None = None,
) -> list[str]:
    """Verify hash, Random Forest contract, class ordering, and inference output.

    ``expected_sha256`` exists for temporary-fixture tests. The command-line
    utility always uses the pinned production checksum.
    """
    if not model_path.is_file():
        raise ArtifactVerificationError(f"Alphabet model file not found: {model_path}")
    metadata = _read_metadata(metadata_path)

    actual_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    required_sha256 = expected_sha256 or EXPECTED_SHA256
    if not re.fullmatch(r"[0-9a-fA-F]{64}", required_sha256):
        raise ArtifactVerificationError("Expected SHA-256 must contain exactly 64 hex digits.")
    if actual_sha256.lower() != required_sha256.lower():
        raise ArtifactVerificationError(
            f"Model SHA-256 mismatch: expected {required_sha256.lower()}, "
            f"got {actual_sha256}."
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ArtifactVerificationError(f"Could not load model {model_path}: {exc}") from exc
    if not isinstance(model, RandomForestClassifier):
        raise ArtifactVerificationError(
            f"Expected RandomForestClassifier, got {type(model).__name__}."
        )
    if getattr(model, "n_estimators", None) != EXPECTED_ESTIMATORS:
        raise ArtifactVerificationError(
            f"Expected {EXPECTED_ESTIMATORS} estimators, got "
            f"{getattr(model, 'n_estimators', None)}."
        )
    if len(getattr(model, "estimators_", [])) != EXPECTED_ESTIMATORS:
        raise ArtifactVerificationError(
            f"Expected {EXPECTED_ESTIMATORS} fitted trees, got "
            f"{len(getattr(model, 'estimators_', []))}."
        )
    if getattr(model, "n_features_in_", None) != FEATURE_COUNT:
        raise ArtifactVerificationError(
            f"Expected {FEATURE_COUNT} input features, got "
            f"{getattr(model, 'n_features_in_', None)}."
        )

    classes = [str(label) for label in getattr(model, "classes_", [])]
    if classes != list(metadata["labels"]):
        raise ArtifactVerificationError(
            "Model classes must match metadata ordering A through Z."
        )
    if len(classes) != len(LABELS):
        raise ArtifactVerificationError(f"Expected 26 model classes, got {len(classes)}.")

    sample = np.zeros((1, FEATURE_COUNT), dtype=np.float32)
    try:
        probabilities = np.asarray(model.predict_proba(sample))
        predictions = np.asarray(model.predict(sample))
    except Exception as exc:
        raise ArtifactVerificationError(f"Model inference failed: {exc}") from exc

    if probabilities.shape != (1, len(LABELS)):
        raise ArtifactVerificationError(
            f"predict_proba must return shape (1, 26), got {probabilities.shape}."
        )
    row = probabilities[0]
    if not np.isfinite(row).all():
        raise ArtifactVerificationError("predict_proba returned non-finite probabilities.")
    if np.any(row < 0.0) or np.any(row > 1.0):
        raise ArtifactVerificationError("predict_proba values must be between 0 and 1.")
    if not np.isclose(float(row.sum()), 1.0, atol=1e-6, rtol=0.0):
        raise ArtifactVerificationError(
            f"predict_proba values are not normalized; sum={float(row.sum()):.8g}."
        )
    if predictions.shape != (1,) or str(predictions[0]) not in LABELS:
        raise ArtifactVerificationError(
            f"predict must return one A-Z label, got {predictions.tolist()}."
        )
    expected_prediction = classes[int(np.argmax(row))]
    if str(predictions[0]) != expected_prediction:
        raise ArtifactVerificationError(
            f"predict label {predictions[0]!r} does not match the highest-probability "
            f"class {expected_prediction!r}."
        )

    return [
        f"PASS model file exists: {model_path}",
        f"PASS SHA-256 matches pinned artifact: {actual_sha256}",
        f"PASS loaded sklearn.ensemble.RandomForestClassifier ({EXPECTED_ESTIMATORS} trees)",
        f"PASS feature count={FEATURE_COUNT}; class count={len(classes)}; classes=A-Z",
        "PASS metadata ordering, label maps, feature count, and Right handedness",
        "PASS predict_proba returns 26 finite normalized probabilities",
        f"PASS prediction is valid A-Z label: {predictions[0]}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    try:
        results = verify_alphabet_model_artifacts(
            model_path=args.model,
            metadata_path=args.metadata,
        )
    except (ArtifactVerificationError, OSError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(result)
    print("PASS all alphabet model artifact checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
