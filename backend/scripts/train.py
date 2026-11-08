"""Train and evaluate the grouped ASL alphabet landmark classifier."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.bootstrap_dataset import FEATURE_COUNT, LETTERS  # noqa: E402

DEFAULT_DATASET = ROOT / "backend" / "data" / "samples.csv"
DEFAULT_METADATA = ROOT / "backend" / "models" / "letters_meta.json"
DEFAULT_MODELS_DIR = ROOT / "backend" / "models"
SEED = 42
AUGMENTATION_RATIO = 10.0
TRAINABLE_LETTERS = list(LETTERS)


def load_feature_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the grouped label, feature, and group columns."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        expected_columns = {"label", "group", *(f"f{index}" for index in range(FEATURE_COUNT))}
        if reader.fieldnames is None or set(reader.fieldnames) != expected_columns:
            raise ValueError("Feature CSV must contain label, group, and f0 through f62 columns.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Feature dataset contains no rows: {path}")

    labels = np.asarray([row["label"] for row in rows], dtype=str)
    groups = np.asarray([row["group"] for row in rows], dtype=str)
    unknown_labels = sorted(set(labels) - set(LETTERS))
    missing_labels = sorted(set(LETTERS) - set(labels))
    if unknown_labels or missing_labels:
        raise ValueError(
            f"Feature labels must contain exactly A-Z; unknown={unknown_labels}, "
            f"missing={missing_labels}."
        )
    if np.any(groups == ""):
        raise ValueError("Group identifiers must be non-empty.")

    try:
        features = np.asarray(
            [[float(row[f"f{index}"]) for index in range(FEATURE_COUNT)] for row in rows],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Feature CSV contains invalid numeric values.") from exc
    if features.shape != (len(rows), FEATURE_COUNT) or not np.isfinite(features).all():
        raise ValueError("Feature CSV must contain finite values for all 63 features.")
    return features, labels, groups


def augment_features(features: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Create one lightly rotated, scaled, and noise-perturbed copy per sample."""
    if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
        raise ValueError(f"Features must have shape (N, {FEATURE_COUNT}).")
    rng = np.random.default_rng(seed)
    points = features.reshape(-1, 21, 3).copy()
    angles = np.deg2rad(rng.uniform(-10.0, 10.0, size=(len(points), 1)))
    cosines = np.cos(angles)
    sines = np.sin(angles)
    x_values = points[:, :, 0].copy()
    y_values = points[:, :, 1].copy()
    points[:, :, 0] = cosines * x_values - sines * y_values
    points[:, :, 1] = sines * x_values + cosines * y_values
    scales = rng.uniform(0.95, 1.05, size=(len(points), 1, 1))
    points *= scales
    points += rng.normal(0.0, 0.01, size=points.shape)
    return points.reshape(-1, FEATURE_COUNT)


def _fit_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    augment: bool,
    seed: int = SEED,
) -> RandomForestClassifier:
    """Fit the specified deterministic 300-tree classifier."""
    if augment:
        augmented = augment_features(features, seed=seed)
        features = np.concatenate((features, augmented), axis=0)
        labels = np.concatenate((labels, labels), axis=0)
    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    return classifier.fit(features, labels)


def _evaluate(
    classifier: RandomForestClassifier,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Calculate overall, per-letter, and confusion-matrix results."""
    predictions = classifier.predict(features)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=LETTERS, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "per_letter": {
            letter: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, letter in enumerate(LETTERS)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=LETTERS).tolist(),
    }


def _plot_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    """Write the selected model's labeled 26-by-26 confusion matrix."""
    figure, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(LETTERS)),
        yticks=np.arange(len(LETTERS)),
        xticklabels=LETTERS,
        yticklabels=LETTERS,
        xlabel="Predicted letter",
        ylabel="True letter",
        title="ASL Alphabet Confusion Matrix",
    )
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(
    dataset_path: Path = DEFAULT_DATASET,
    metadata_path: Path = DEFAULT_METADATA,
    models_dir: Path = DEFAULT_MODELS_DIR,
    augment: bool = True,
    holdout_group: str | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compare augmentation modes and save the selected grouped-split model and metrics."""
    output_paths = [
        models_dir / "asl_model.joblib",
        models_dir / "metrics.json",
        models_dir / "confusion_matrix.png",
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing model artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("label_to_id") != {letter: index for index, letter in enumerate(LETTERS)}:
        raise ValueError("letters_meta.json label mapping does not match A-Z order.")

    features, labels, groups = load_feature_dataset(dataset_path)
    sample_counts = {letter: int(np.count_nonzero(labels == letter)) for letter in LETTERS}
    for letter, count in sample_counts.items():
        if count < 50:
            print(f"WARNING: {letter} has only {count} feature rows (<50).")

    holdout_mask = np.zeros(len(labels), dtype=bool)
    if holdout_group is not None:
        if not holdout_group.startswith("live-"):
            raise ValueError("A team-member holdout group must use the live-NAME form.")
        holdout_mask = groups == holdout_group
        if not holdout_mask.any():
            raise ValueError(f"Holdout group {holdout_group!r} is not present in the dataset.")
    development_mask = ~holdout_mask
    development_groups = groups[development_mask]
    development_indices = np.flatnonzero(development_mask)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    try:
        train_relative, validation_relative = next(
            splitter.split(features[development_mask], labels[development_mask], development_groups)
        )
    except ValueError as exc:
        raise ValueError("At least two distinct development groups are required for an 80/20 split.") from exc
    train_indices = development_indices[train_relative]
    validation_indices = development_indices[validation_relative]
    train_groups = sorted(set(groups[train_indices]))
    validation_groups = sorted(set(groups[validation_indices]))
    if set(train_groups) & set(validation_groups):
        raise AssertionError("Group leakage detected between training and validation partitions.")
    if set(labels[train_indices]) != set(LETTERS):
        missing = sorted(set(LETTERS) - set(labels[train_indices]))
        raise ValueError(f"Training partition is missing letters: {missing}.")

    comparison: dict[str, Any] = {}
    classifiers: dict[bool, RandomForestClassifier] = {}
    for use_augmentation in (False, True):
        classifier = _fit_classifier(
            features[train_indices],
            labels[train_indices],
            augment=use_augmentation,
            seed=seed,
        )
        key = "augmented" if use_augmentation else "unaugmented"
        comparison[key] = _evaluate(
            classifier,
            features[validation_indices],
            labels[validation_indices],
        )
        classifiers[use_augmentation] = classifier

    selected_classifier = classifiers[augment]
    live_evaluation = None
    live_training_groups: list[str] = []
    if holdout_group is not None:
        live_indices = np.flatnonzero(holdout_mask)
        non_live_indices = np.flatnonzero(development_mask)
        live_classifier = _fit_classifier(
            features[non_live_indices],
            labels[non_live_indices],
            augment=augment,
            seed=seed,
        )
        live_evaluation = _evaluate(
            live_classifier,
            features[live_indices],
            labels[live_indices],
        )
        live_training_groups = sorted(set(groups[non_live_indices]))

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_classifier, output_paths[0])
    _plot_confusion_matrix(comparison["augmented" if augment else "unaugmented"]["confusion_matrix"], output_paths[2])
    metrics = {
        "dataset": str(dataset_path),
        "seed": seed,
        "model": "RandomForestClassifier",
        "n_estimators": 300,
        "augmentation_selected": augment,
        "overall_accuracy": comparison["augmented" if augment else "unaugmented"]["accuracy"],
        "sample_count": len(labels),
        "sample_count_per_letter": sample_counts,
        "training_sample_count": len(train_indices),
        "evaluation_sample_count": len(validation_indices),
        "training_groups": train_groups,
        "evaluation_groups": validation_groups,
        "group_split": {
            "method": "GroupShuffleSplit",
            "train_fraction": 0.8,
            "evaluation_fraction": 0.2,
            "group_disjoint": True,
            "group_definition": "source image index block of 100; not signer identity",
            "signer_independence_established": False,
        },
        "augmentation_comparison": comparison,
        "selected_per_letter": comparison["augmented" if augment else "unaugmented"]["per_letter"],
        "selected_confusion_matrix": comparison[
            "augmented" if augment else "unaugmented"
        ]["confusion_matrix"],
        "holdout_group": holdout_group,
        "live_holdout_sample_count": int(np.count_nonzero(holdout_mask)),
        "holdout_training_groups": live_training_groups,
        "live_holdout_evaluation": live_evaluation,
    }
    output_paths[1].write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _parse_args() -> argparse.Namespace:
    """Parse training and evaluation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--holdout-group", help="Separate team-member group, e.g. live-AYUSH.")
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument("--augment", dest="augment", action="store_true")
    augmentation.add_argument("--no-augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    """Run training and report the selected evaluation accuracy."""
    args = _parse_args()
    metrics = train(
        dataset_path=args.dataset,
        metadata_path=args.metadata,
        models_dir=args.models_dir,
        augment=args.augment,
        holdout_group=args.holdout_group,
        seed=args.seed,
    )
    print(f"Selected model accuracy: {metrics['overall_accuracy']:.4f}")
    print(
        "Augmentation comparison: "
        f"no-augment={metrics['augmentation_comparison']['unaugmented']['accuracy']:.4f}, "
        f"augment={metrics['augmentation_comparison']['augmented']['accuracy']:.4f}"
    )
    print(
        f"Groups: train={len(metrics['training_groups'])}, "
        f"evaluation={len(metrics['evaluation_groups'])}; group IDs are image blocks, "
        "not signer IDs."
    )
    if metrics["live_holdout_evaluation"] is not None:
        print(
            f"Live holdout {metrics['holdout_group']} accuracy: "
            f"{metrics['live_holdout_evaluation']['accuracy']:.4f}"
        )
    print("Per-letter precision / recall / F1:")
    selected_per_letter = metrics["selected_per_letter"]
    for letter in LETTERS:
        result = selected_per_letter[letter]
        print(
            f"  {letter}: {result['precision']:.4f} / "
            f"{result['recall']:.4f} / {result['f1']:.4f}"
        )
    print(f"Model: {args.models_dir / 'asl_model.joblib'}")
    print(f"Metrics: {args.models_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
