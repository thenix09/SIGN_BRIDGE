"""Extract grouped ASL alphabet landmark features from the Kaggle image dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Protocol
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.domain.hand_frame import HandFrame, canonicalise_landmarks  # noqa: E402

LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IGNORED_LABELS = frozenset({"del", "nothing", "space"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
FEATURE_COUNT = 63
DEFAULT_IMAGES_DIR = ROOT / "backend" / "data" / "raw" / "asl_alphabet_train"
DEFAULT_OUTPUT = ROOT / "backend" / "data" / "samples.csv"
DEFAULT_REPORT = ROOT / "backend" / "data" / "bootstrap_report.json"
DEFAULT_METADATA = ROOT / "backend" / "models" / "letters_meta.json"
DEFAULT_TASK_MODEL = ROOT / "backend" / "models" / "hand_landmarker.task"
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class HandDetector(Protocol):
    """Image detector interface used by the bootstrap pipeline and its tests."""

    def detect(self, image_path: Path) -> tuple[list[list[float]], str] | None:
        """Return the first detected hand's raw landmarks and handedness label."""


class MediaPipeHandDetector:
    """Run MediaPipe HandLandmarker in static-image mode."""

    def __init__(self, model_path: Path, min_detection_confidence: float = 0.5) -> None:
        """Create a one-hand MediaPipe image detector using a local task model."""
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0 and 1.")
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is required; install requirements-dev.txt.") from exc

        self._cv2 = __import__("cv2")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def detect(self, image_path: Path) -> tuple[list[list[float]], str] | None:
        """Detect one hand in an image, or return None for an unreadable/no-hand image."""
        image = self._cv2.imread(str(image_path))
        if image is None:
            return None
        image = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks or not result.handedness:
            return None
        points = result.hand_landmarks[0]
        handedness = result.handedness[0][0].category_name
        raw = [[point.x, point.y, point.z] for point in points]
        if len(raw) != 21 or handedness not in {"Left", "Right"}:
            return None
        return raw, handedness


def find_images_dir(hint: Path | None = None) -> Path:
    """Find the alphabet image root locally or under a Kaggle input mount."""
    candidates: list[Path] = []
    if hint is not None:
        candidates.append(hint)
    candidates.extend(
        [
            DEFAULT_IMAGES_DIR,
            Path("/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"),
            Path("/kaggle/input/asl-alphabet/asl_alphabet_train"),
            Path("/kaggle/input/asl-alphabet"),
        ]
    )
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(path for path in kaggle_input.iterdir() if path.is_dir())

    for candidate in candidates:
        if _is_image_root(candidate):
            return candidate
        if candidate.is_dir():
            for nested in candidate.rglob("asl_alphabet_train"):
                if _is_image_root(nested):
                    return nested
    raise FileNotFoundError(
        "Could not find folders A-Z under the Kaggle/local ASL Alphabet dataset. "
        "Pass --images-dir pointing to asl_alphabet_train."
    )


def _is_image_root(path: Path) -> bool:
    """Return whether a directory contains at least one folder for each letter."""
    return path.is_dir() and all((path / label).is_dir() for label in LETTERS)


def group_for_image(image_index: int) -> str:
    """Map each consecutive block of 100 original images to one split group."""
    if image_index < 0:
        raise ValueError("Image index must be non-negative.")
    return f"img{image_index // 100}"


def ensure_task_model(model_path: Path) -> Path:
    """Download the official MediaPipe task model when it is not already present."""
    if model_path.is_file() and model_path.stat().st_size:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(HAND_LANDMARKER_URL, timeout=60) as response:
            content = response.read()
    except (OSError, URLError) as exc:
        raise RuntimeError(
            f"Could not download MediaPipe hand model from {HAND_LANDMARKER_URL}: {exc}"
        ) from exc
    if not content:
        raise RuntimeError(f"MediaPipe hand model download was empty: {HAND_LANDMARKER_URL}")
    model_path.write_bytes(content)
    return model_path


def _selected_images(images_dir: Path, per_class: int, seed: int) -> dict[str, list[tuple[int, Path]]]:
    """Select a deterministic subset while retaining each source image's sorted index."""
    rng = random.Random(seed)
    selected: dict[str, list[tuple[int, Path]]] = {}
    for label in LETTERS:
        paths = sorted(
            path
            for path in (images_dir / label).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        indices = sorted(rng.sample(range(len(paths)), min(per_class, len(paths))))
        selected[label] = [(index, paths[index]) for index in indices]
    return selected


def bootstrap_dataset(
    images_dir: Path,
    output_path: Path,
    report_path: Path,
    metadata_path: Path,
    detector: HandDetector,
    per_class: int = 400,
    seed: int = 42,
) -> dict[str, object]:
    """Extract canonicalized feature rows, group identifiers, metadata, and a report."""
    if per_class < 1:
        raise ValueError("per_class must be positive.")
    for path in (output_path, report_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing alphabet artifact: {path}")

    selected = _selected_images(images_dir, per_class, seed)
    counts = {
        label: {
            "tried": len(selected[label]),
            "detected": 0,
            "failed_detections": 0,
            "written": 0,
        }
        for label in LETTERS
    }
    detected_rows: list[tuple[str, str, list[list[float]], str]] = []
    handedness_counts = {"Left": 0, "Right": 0}
    for label in LETTERS:
        for image_index, image_path in selected[label]:
            detection = detector.detect(image_path)
            if detection is None:
                continue
            raw_landmarks, handedness = detection
            if len(raw_landmarks) != 21 or handedness not in handedness_counts:
                continue
            counts[label]["detected"] += 1
            handedness_counts[handedness] += 1
            detected_rows.append(
                (label, group_for_image(image_index), raw_landmarks, handedness)
            )

    if not detected_rows:
        raise ValueError("No hand landmarks were detected; no dataset artifacts were written.")
    canonical_handedness = max(handedness_counts, key=handedness_counts.__getitem__)
    rows: list[tuple[str, str, list[float]]] = []
    for label, group, landmarks, handedness in detected_rows:
        canonical = canonicalise_landmarks(landmarks, handedness, canonical_handedness)
        features = HandFrame.from_raw(canonical).to_feature_vector()
        if len(features) != FEATURE_COUNT or not np.isfinite(features).all():
            continue
        rows.append((label, group, features))
        counts[label]["written"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "group", *(f"f{index}" for index in range(FEATURE_COUNT))]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        for label, group, features in rows:
            writer.writerow([label, group, *features])

    total_tried = sum(entry["tried"] for entry in counts.values())
    total_detected = sum(entry["detected"] for entry in counts.values())
    total_written = sum(entry["written"] for entry in counts.values())
    for entry in counts.values():
        entry["failed_detections"] = entry["tried"] - entry["detected"]
    total_failed_detections = total_tried - total_detected
    missing_classes = [label for label in LETTERS if counts[label]["written"] == 0]
    low_sample_classes = [label for label in LETTERS if counts[label]["written"] < 100]
    metadata = {
        "dataset_name": "Kaggle ASL Alphabet",
        "dataset_url": "https://www.kaggle.com/datasets/grassknoted/asl-alphabet",
        "dataset_license": "GPL-2",
        "labels": list(LETTERS),
        "label_to_id": {label: index for index, label in enumerate(LETTERS)},
        "id_to_label": {str(index): label for index, label in enumerate(LETTERS)},
        "canonical_handedness": canonical_handedness,
        "feature_count": FEATURE_COUNT,
        "feature_layout": "21 wrist-origin, max-distance-normalized landmarks in XYZ order",
        "seed": seed,
        "per_class_limit": per_class,
    }
    report = {
        "dataset_name": metadata["dataset_name"],
        "source_directory": str(images_dir),
        "seed": seed,
        "per_class_limit": per_class,
        "canonical_handedness": canonical_handedness,
        "handedness_counts": handedness_counts,
        "total_tried": total_tried,
        "total_detected": total_detected,
        "total_failed_detections": total_failed_detections,
        "total_written": total_written,
        "detection_rate": total_detected / total_tried if total_tried else 0.0,
        "missing_classes": missing_classes,
        "classes_below_100_rows": low_sample_classes,
        "per_class": counts,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for label in low_sample_classes:
        print(f"WARNING: {label} has only {counts[label]['written']} valid feature rows (<100).")
    if missing_classes:
        print(f"WARNING: no valid feature rows for classes: {', '.join(missing_classes)}")
    return report


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset bootstrap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--task-model", type=Path, default=DEFAULT_TASK_MODEL)
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the bootstrap CLI using MediaPipe static-image inference."""
    args = _parse_args()
    images_dir = find_images_dir(args.images_dir)
    model_path = ensure_task_model(args.task_model)
    detector = MediaPipeHandDetector(model_path, args.min_detection_confidence)
    try:
        report = bootstrap_dataset(
            images_dir,
            args.output,
            args.report,
            args.metadata,
            detector,
            per_class=args.per_class,
            seed=args.seed,
        )
    finally:
        detector.close()
    print(
        f"Bootstrap complete: {report['total_written']} feature rows from "
        f"{report['total_tried']} selected images."
    )


if __name__ == "__main__":
    main()
