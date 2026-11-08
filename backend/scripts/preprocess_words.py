"""SignBridge ASL Words Mode Preprocessing Pipeline.

Extracts normalized (32, 104) feature tensors from raw Kaggle Parquet sequences
using Person A's exact domain implementation in backend/app/domain/sequence.py.

Can be run locally on downloaded subsets or in a Kaggle Notebook environment
with the full asl-signs dataset attached.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is on sys.path for direct script execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from backend.app.core.exceptions import InvalidLandmarkError
from backend.app.domain.sequence import (
    LandmarkFrame,
    LandmarkWindow,
    N_FEATURES,
    N_HAND,
    N_POSE,
    POSE_INDICES,
    T,
)

GIT_COMMIT_HASH = "22a2b1c42175300e62e30eff11e637bcce36adff"

TRAIN_SIGNERS = [
    2044, 4718, 16069, 18796, 22343, 25571, 27610,
    28656, 29302, 30680, 34503, 36257, 37055, 62590,
]
VAL_SIGNERS = [26734, 53618, 55372]
TEST_SIGNERS = [32319, 37779, 49445, 61333]


def find_dataset_root(hint_path: Path | None = None) -> Path:
    """Automatically discover the root directory containing Kaggle asl-signs data."""
    candidates: list[Path] = []
    if hint_path:
        candidates.append(Path(hint_path))

    # Common Kaggle dataset mount locations
    candidates.extend([
        Path("/kaggle/input/asl-signs"),
        Path("/kaggle/input/isolated-sign-language-recognition"),
        ROOT / "backend" / "data" / "raw",
    ])

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.iterdir():
            if p.is_dir() and ((p / "train_landmark_files").exists() or (p / "train.csv").exists()):
                candidates.insert(0, p)

    for cand in candidates:
        if cand.exists() and (cand / "train_landmark_files").exists():
            return cand

    raise FileNotFoundError(
        "Could not automatically locate the Kaggle asl-signs dataset directory. "
        "Please provide --dataset-root pointing to the directory containing 'train_landmark_files'."
    )


def parquet_to_tensor(file_path: Path | str) -> np.ndarray:
    """Read a Parquet file and convert to normalized, canonicalized (32, 104) tensor.

    Steps:
        1. Read coordinates discarding 468 facial mesh landmarks and z coordinates.
        2. Group frames and verify required pose indices.
        3. Convert missing or NaN hands to None (absent hands).
        4. Construct LandmarkFrame instances using Person A's domain model.
        5. Pass frames through LandmarkWindow.to_tensor() (shoulder norm, resample to 32,
           mirror canonicalization).
        6. Validate shape (32, 104), dtype float32, and finiteness.
    """
    path_obj = Path(file_path)
    if not path_obj.exists() or path_obj.stat().st_size == 0:
        raise FileNotFoundError(f"Parquet file missing or empty: {file_path}")

    # Read required landmark columns, filtering out face landmarks
    table = pq.read_table(
        str(path_obj),
        columns=["frame", "type", "landmark_index", "x", "y"],
        filters=[("type", "!=", "face")],
    )
    df = table.to_pandas()
    if df.empty:
        raise ValueError(f"No non-face landmarks found in: {file_path}")

    unique_frames = sorted(df["frame"].unique())
    if not unique_frames:
        raise ValueError(f"No frames in parquet file: {file_path}")

    # Extract pose and verify 9 points
    pose_df = df[(df["type"] == "pose") & (df["landmark_index"].isin(POSE_INDICES))]
    pose_x = pose_df.pivot(index="frame", columns="landmark_index", values="x")[list(POSE_INDICES)]
    pose_y = pose_df.pivot(index="frame", columns="landmark_index", values="y")[list(POSE_INDICES)]

    # Extract left and right hands
    lh_df = df[df["type"] == "left_hand"]
    lh_x = lh_df.pivot(index="frame", columns="landmark_index", values="x")[list(range(N_HAND))]
    lh_y = lh_df.pivot(index="frame", columns="landmark_index", values="y")[list(range(N_HAND))]

    rh_df = df[df["type"] == "right_hand"]
    rh_x = rh_df.pivot(index="frame", columns="landmark_index", values="x")[list(range(N_HAND))]
    rh_y = rh_df.pivot(index="frame", columns="landmark_index", values="y")[list(range(N_HAND))]

    frames: list[LandmarkFrame] = []

    for frame_id in unique_frames:
        # Check pose coordinates exist and are finite
        if frame_id not in pose_x.index or frame_id not in pose_y.index:
            raise ValueError(f"Frame {frame_id} missing pose in {file_path}")

        px_row = pose_x.loc[frame_id].values
        py_row = pose_y.loc[frame_id].values
        if np.isnan(px_row).any() or np.isnan(py_row).any():
            raise ValueError(f"NaN pose coordinates in frame {frame_id} of {file_path}")

        pose_coords = tuple((float(x), float(y)) for x, y in zip(px_row, py_row))

        # Check left hand: absent if missing or any coordinate is NaN
        left_hand: tuple[tuple[float, float], ...] | None = None
        if frame_id in lh_x.index and frame_id in lh_y.index:
            lx_row = lh_x.loc[frame_id].values
            ly_row = lh_y.loc[frame_id].values
            if not np.isnan(lx_row).any() and not np.isnan(ly_row).any():
                left_hand = tuple((float(x), float(y)) for x, y in zip(lx_row, ly_row))

        # Check right hand: absent if missing or any coordinate is NaN
        right_hand: tuple[tuple[float, float], ...] | None = None
        if frame_id in rh_x.index and frame_id in rh_y.index:
            rx_row = rh_x.loc[frame_id].values
            ry_row = rh_y.loc[frame_id].values
            if not np.isnan(rx_row).any() and not np.isnan(ry_row).any():
                right_hand = tuple((float(x), float(y)) for x, y in zip(rx_row, ry_row))

        frames.append(
            LandmarkFrame(
                pose=pose_coords,
                left_hand=left_hand,
                right_hand=right_hand,
            )
        )

    # Transform through LandmarkWindow
    window = LandmarkWindow(frames)
    tensor = window.to_tensor()

    # Validation assertions
    if tensor.shape != (T, N_FEATURES):
        raise ValueError(f"Expected shape ({T}, {N_FEATURES}), got {tensor.shape} in {file_path}")
    if tensor.dtype != np.float32:
        raise ValueError(f"Expected float32 dtype, got {tensor.dtype} in {file_path}")
    if not np.isfinite(tensor).all():
        raise ValueError(f"Non-finite values detected in tensor for {file_path}")

    return tensor


def load_vocabulary(vocab_path: Path) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Load the 30 words and create deterministic label mappings."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = [item["label"] for item in data["words"]]
    # Sort vocabulary alphabetically for standard deterministic class indexing
    sorted_words = sorted(list(dict.fromkeys(words)))
    if len(sorted_words) != 30:
        raise ValueError(f"Expected 30 unique vocabulary words, got {len(sorted_words)}")

    label_to_id = {word: idx for idx, word in enumerate(sorted_words)}
    id_to_label = {idx: word for idx, word in enumerate(sorted_words)}
    return sorted_words, label_to_id, id_to_label


def process_manifest(
    manifest_csv: Path,
    dataset_root: Path,
    output_dir: Path,
    vocab_path: Path,
    limit: int | None = None,
    checkpoint_every: int = 100,
) -> tuple[Path, Path]:
    """Process manifest rows into word_dataset.npz and meta.json with validation."""
    print("=" * 70)
    print("SignBridge Words Mode Preprocessing Pipeline")
    print(f"Manifest:       {manifest_csv}")
    print(f"Dataset Root:   {dataset_root}")
    print(f"Output Dir:     {output_dir}")
    print(f"Commit Hash:    {GIT_COMMIT_HASH}")
    print("=" * 70)

    manifest_df = pd.read_csv(manifest_csv)
    total_manifest = len(manifest_df)
    print(f"Loaded manifest with {total_manifest} entries.")

    if limit is not None and limit > 0:
        manifest_df = manifest_df.iloc[:limit].copy()
        print(f"** Subsetting to first {len(manifest_df)} entries for local testing **")

    sorted_words, label_to_id, id_to_label = load_vocabulary(vocab_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(manifest_df)
    X = np.zeros((n_samples, T, N_FEATURES), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    signs = np.empty(n_samples, dtype=object)
    splits = np.empty(n_samples, dtype=object)
    participant_ids = np.zeros(n_samples, dtype=np.int64)
    sequence_ids = np.zeros(n_samples, dtype=np.int64)
    paths = np.empty(n_samples, dtype=object)

    start_time = time.time()
    last_report_time = start_time

    for idx, row in enumerate(manifest_df.itertuples()):
        rel_path = row.path
        abs_path = dataset_root / rel_path

        try:
            tensor = parquet_to_tensor(abs_path)
        except Exception as exc:
            err_msg = f"FATAL PARSING FAILURE at row {idx} for path '{rel_path}': {exc}"
            print(f"\n[ERROR] {err_msg}", file=sys.stderr)
            raise RuntimeError(err_msg) from exc

        sign = str(row.sign)
        if sign not in label_to_id:
            raise ValueError(f"Unknown sign '{sign}' in row {idx} (path: {rel_path})")

        X[idx] = tensor
        y[idx] = label_to_id[sign]
        signs[idx] = sign
        splits[idx] = str(row.split)
        participant_ids[idx] = int(row.participant_id)
        sequence_ids[idx] = int(row.sequence_id)
        paths[idx] = rel_path

        # Progress reporting
        if (idx + 1) % checkpoint_every == 0 or (idx + 1) == n_samples:
            now = time.time()
            elapsed = now - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"Progress: [{idx + 1:5d} / {n_samples:5d}] "
                f"({(idx + 1) / n_samples * 100:5.1f}%) | "
                f"Elapsed: {elapsed:5.1f}s | Speed: {rate:5.1f} seq/s"
            )

            # Save checkpoint slice
            cp_file = checkpoint_dir / f"checkpoint_batch_{idx + 1}.npz"
            np.savez_compressed(
                cp_file,
                X_batch=X[: idx + 1],
                y_batch=y[: idx + 1],
                paths_batch=paths[: idx + 1],
            )

    print("\nSequence processing complete. Running rigorous quality assertions...")

    # Quality Checks
    if limit is None:
        assert len(X) == 6000, f"Expected exactly 6000 sequences, got {len(X)}"
        assert len(np.unique(paths)) == 6000, "Duplicate paths found in processed dataset!"

        # Check split quotas
        train_count = np.sum(splits == "train")
        val_count = np.sum(splits == "val")
        test_count = np.sum(splits == "test")
        print(f"Split counts: train={train_count}, val={val_count}, test={test_count}")
        assert train_count == 4200, f"Expected 4200 train samples, got {train_count}"
        assert val_count == 900, f"Expected 900 val samples, got {val_count}"
        assert test_count == 900, f"Expected 900 test samples, got {test_count}"

        # Check signer isolation
        train_s = set(participant_ids[splits == "train"])
        val_s = set(participant_ids[splits == "val"])
        test_s = set(participant_ids[splits == "test"])
        assert train_s.isdisjoint(val_s), "Signer leakage: train and val overlap!"
        assert train_s.isdisjoint(test_s), "Signer leakage: train and test overlap!"
        assert val_s.isdisjoint(test_s), "Signer leakage: val and test overlap!"

        # Check all 30 words have 200 sequences
        for w in sorted_words:
            w_count = np.sum(signs == w)
            assert w_count == 200, f"Word '{w}' has {w_count} sequences, expected 200"

    assert X.shape == (n_samples, T, N_FEATURES), f"Shape mismatch: {X.shape}"
    assert X.dtype == np.float32, f"Dtype mismatch: {X.dtype}"
    assert np.isfinite(X).all(), "Non-finite values found in final X tensor!"

    # Save to atomic temporary files
    temp_npz = output_dir / "word_dataset.tmp.npz"
    final_npz = output_dir / "word_dataset.npz"

    print(f"Writing compressed dataset to temporary file: {temp_npz}")
    np.savez_compressed(
        temp_npz,
        X=X,
        y=y,
        label_id=y,
        sign=signs.astype(str),
        split=splits.astype(str),
        participant_id=participant_ids,
        sequence_id=sequence_ids,
        path=paths.astype(str),
    )

    # Atomic rename
    temp_npz.replace(final_npz)
    print(f"Final dataset atomically finalized: {final_npz} ({final_npz.stat().st_size / 1e6:.2f} MB)")

    # Prepare meta.json
    metadata: dict[str, Any] = {
        "dataset_name": "SignBridge ASL Words Mode Preprocessed Tensors",
        "description": "Preprocessed isolated sign tensors with dominant-hand canonicalisation and shoulder normalization.",
        "git_commit": GIT_COMMIT_HASH,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_sequences": int(n_samples),
        "split_counts": {
            "train": int(np.sum(splits == "train")),
            "val": int(np.sum(splits == "val")),
            "test": int(np.sum(splits == "test")),
        },
        "tensor_spec": {
            "T": T,
            "N_FEATURES": N_FEATURES,
            "N_POSE": N_POSE,
            "N_HAND": N_HAND,
            "pose_indices": list(POSE_INDICES),
            "feature_layout": "18 pose (x, y) + 42 left hand (x, y) + 42 right hand (x, y) + 2 presence flags",
            "dtype": "float32",
        },
        "signer_splits": {
            "train": TRAIN_SIGNERS,
            "val": VAL_SIGNERS,
            "test": TEST_SIGNERS,
        },
        "vocabulary": sorted_words,
        "label_to_id": label_to_id,
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
    }

    temp_meta = output_dir / "meta.tmp.json"
    final_meta = output_dir / "meta.json"
    with open(temp_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    temp_meta.replace(final_meta)
    print(f"Final metadata atomically finalized: {final_meta}")

    print("=" * 70)
    print(">>> PREPROCESSING & VALIDATION SUCCESSFUL <<<")
    print("=" * 70)
    return final_npz, final_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="SignBridge Words Mode Preprocessing Pipeline.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "backend" / "data" / "processed" / "word_manifest.csv",
        help="Path to approved 6,000 sequence manifest CSV",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root path containing train_landmark_files (auto-detected if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "backend" / "data" / "processed",
        help="Directory to save word_dataset.npz and meta.json",
    )
    parser.add_argument(
        "--vocab-file",
        type=Path,
        default=ROOT / "backend" / "data" / "vocabulary.json",
        help="Path to vocabulary.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of sequences to process (for local testing)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Progress reporting and checkpoint frequency",
    )
    args = parser.parse_args()

    dataset_root = find_dataset_root(args.dataset_root)
    process_manifest(
        manifest_csv=args.manifest,
        dataset_root=dataset_root,
        output_dir=args.output_dir,
        vocab_path=args.vocab_file,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
