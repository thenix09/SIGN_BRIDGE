import csv
import json
from pathlib import Path

import numpy as np

from backend.app.domain.hand_frame import HandFrame, canonicalise_landmarks
from backend.scripts.bootstrap_dataset import (
    FEATURE_COUNT,
    LETTERS,
    bootstrap_dataset,
    group_for_image,
)
from backend.scripts.train import train


def _raw_hand(offset: float = 0.0) -> list[list[float]]:
    return [
        [0.2 + offset + index * 0.01, 0.3 + index * 0.015, index * 0.005]
        for index in range(21)
    ]


class FilenameHandDetector:
    """Mock detector returning no hand for selected filenames."""

    def detect(self, image_path: Path) -> tuple[list[list[float]], str] | None:
        if image_path.stem.endswith("missing"):
            return None
        return _raw_hand(), "Right"


def test_hand_frame_feature_count_normalization_and_mirroring() -> None:
    raw = _raw_hand()
    right_features = HandFrame.from_raw(raw).to_feature_vector()
    mirrored = [[1.0 - x, y, z] for x, y, z in raw]
    canonical_left = canonicalise_landmarks(mirrored, "Left", "Right")
    left_features = HandFrame.from_raw(canonical_left).to_feature_vector()

    assert len(right_features) == FEATURE_COUNT
    points = np.asarray(right_features).reshape(21, 3)
    assert np.isclose(np.linalg.norm(points[0]), 0.0)
    assert np.isclose(np.linalg.norm(points, axis=1).max(), 1.0)
    assert np.allclose(right_features, left_features, atol=1e-7)


def test_group_identifiers_use_adjacent_original_image_blocks() -> None:
    assert group_for_image(0) == "img0"
    assert group_for_image(99) == "img0"
    assert group_for_image(100) == "img1"
    assert group_for_image(299) == "img2"


def test_bootstrap_labels_groups_and_missing_hand_detection(tmp_path: Path) -> None:
    images = tmp_path / "asl_alphabet_train"
    for label in LETTERS:
        class_dir = images / label
        class_dir.mkdir(parents=True)
    for index in range(101):
        (images / "A" / f"{index:03d}.jpg").touch()
    (images / "A" / "101-missing.jpg").touch()

    output = tmp_path / "samples.csv"
    report_path = tmp_path / "bootstrap_report.json"
    metadata_path = tmp_path / "letters_meta.json"
    report = bootstrap_dataset(
        images,
        output,
        report_path,
        metadata_path,
        FilenameHandDetector(),
        per_class=102,
        seed=7,
    )

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report_file = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["total_tried"] == 102
    assert report["total_detected"] == 101
    assert report["total_written"] == 101
    assert report["missing_classes"] == list(LETTERS[1:])
    assert {row["label"] for row in rows} == {"A"}
    assert {row["group"] for row in rows} == {"img0", "img1"}
    assert all(len([row[f"f{i}"] for i in range(FEATURE_COUNT)]) == FEATURE_COUNT for row in rows)
    assert metadata["canonical_handedness"] == "Right"
    assert metadata["label_to_id"] == {letter: index for index, letter in enumerate(LETTERS)}
    assert report_file["per_class"]["A"]["detected"] == 101
    assert report_file["per_class"]["A"]["written"] == 101


def test_training_predicts_all_letters_and_reports_grouped_augmentation(tmp_path: Path) -> None:
    dataset_path = tmp_path / "samples.csv"
    metadata_path = tmp_path / "letters_meta.json"
    models_dir = tmp_path / "models"
    metadata_path.write_text(
        json.dumps(
            {
                "labels": list(LETTERS),
                "label_to_id": {letter: index for index, letter in enumerate(LETTERS)},
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(42)
    with dataset_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "group", *(f"f{i}" for i in range(FEATURE_COUNT))])
        for group_index in range(4):
            for label_index, label in enumerate(LETTERS):
                for sample_index in range(2):
                    features = np.zeros(FEATURE_COUNT)
                    features[label_index] = 1.0
                    features += rng.normal(0.0, 0.001, size=FEATURE_COUNT)
                    writer.writerow([label, f"img{group_index}", *features])
        for label_index, label in enumerate(LETTERS):
            features = np.zeros(FEATURE_COUNT)
            features[label_index] = 1.0
            features += rng.normal(0.0, 0.001, size=FEATURE_COUNT)
            writer.writerow([label, "live-AYUSH", *features])

    metrics = train(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        models_dir=models_dir,
        augment=False,
        holdout_group="live-AYUSH",
        seed=19,
    )

    import joblib

    classifier = joblib.load(models_dir / "asl_model.joblib")
    predictions = classifier.predict(np.eye(26, FEATURE_COUNT))
    assert set(predictions) == set(LETTERS)
    assert metrics["group_split"]["group_disjoint"] is True
    assert metrics["group_split"]["group_definition"] != "signer identity"
    assert metrics["group_split"]["signer_independence_established"] is False
    assert set(metrics["augmentation_comparison"]) == {"augmented", "unaugmented"}
    assert len(metrics["sample_count_per_letter"]) == 26
    assert metrics["holdout_group"] == "live-AYUSH"
    assert metrics["live_holdout_sample_count"] == 26
    assert metrics["live_holdout_evaluation"]["accuracy"] >= 0.0
    assert (models_dir / "metrics.json").is_file()
    assert (models_dir / "confusion_matrix.png").is_file()
