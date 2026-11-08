"""Validation and loading for the approved 30-word training dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

EXPECTED_SPLIT_COUNTS = {"train": 4200, "val": 900, "test": 900}
EXPECTED_SPLIT_LABEL_COUNTS = {"train": 140, "val": 30, "test": 30}
EXPECTED_TOTAL = sum(EXPECTED_SPLIT_COUNTS.values())
EXPECTED_SHAPE = (EXPECTED_TOTAL, 32, 104)
REQUIRED_ARRAYS = {
    "X",
    "y",
    "label_id",
    "sign",
    "split",
    "participant_id",
}


class WordDataset(NamedTuple):
    """Validated tensors, grouped by split, and their approved label order."""

    splits: dict[str, tuple[np.ndarray, np.ndarray]]
    vocabulary: list[str]


def load_word_dataset(dataset_path: Path, meta_path: Path) -> WordDataset:
    """Load the approved tensors only after validating their shape and split metadata."""
    with meta_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError("Dataset metadata must be a JSON object.")

    vocabulary = metadata.get("vocabulary")
    if (
        not isinstance(vocabulary, list)
        or len(vocabulary) != 30
        or not all(isinstance(word, str) and word for word in vocabulary)
    ):
        raise ValueError("Metadata must define 30 unique, non-empty vocabulary labels.")
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Metadata must define 30 unique, non-empty vocabulary labels.")

    expected_label_to_id = {word: index for index, word in enumerate(vocabulary)}
    if metadata.get("label_to_id") != expected_label_to_id:
        raise ValueError("Metadata label_to_id does not match the approved vocabulary order.")
    if metadata.get("total_sequences") != EXPECTED_TOTAL:
        raise ValueError(f"Metadata must declare exactly {EXPECTED_TOTAL} sequences.")
    if metadata.get("split_counts") != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Metadata split_counts must be {EXPECTED_SPLIT_COUNTS}.")

    tensor_spec = metadata.get("tensor_spec", {})
    if (
        not isinstance(tensor_spec, dict)
        or tensor_spec.get("T") != 32
        or tensor_spec.get("N_FEATURES") != 104
    ):
        raise ValueError("Metadata tensor specification must be 32 frames by 104 features.")

    signer_splits = metadata.get("signer_splits")
    if not isinstance(signer_splits, dict) or set(signer_splits) != set(EXPECTED_SPLIT_COUNTS):
        raise ValueError("Metadata must declare signer IDs for train, val, and test.")
    expected_signers: dict[str, set[int]] = {}
    for split_name, signer_ids in signer_splits.items():
        if not isinstance(signer_ids, list) or not signer_ids:
            raise ValueError(f"Metadata signer list for {split_name!r} must be non-empty.")
        expected_signers[split_name] = {int(signer_id) for signer_id in signer_ids}
        if len(expected_signers[split_name]) != len(signer_ids):
            raise ValueError(f"Metadata contains duplicate signer IDs in {split_name!r}.")
    if any(
        expected_signers[left] & expected_signers[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Metadata signer IDs overlap across splits.")

    with np.load(dataset_path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS - set(archive.files)
        if missing:
            raise ValueError(f"Dataset is missing required arrays: {sorted(missing)}.")
        arrays = {name: archive[name] for name in REQUIRED_ARRAYS}

    X = arrays["X"]
    y = arrays["y"]
    label_id = arrays["label_id"]
    signs = arrays["sign"]
    split_names = arrays["split"]
    participant_ids = arrays["participant_id"]

    if X.shape != EXPECTED_SHAPE or X.dtype != np.float32:
        raise ValueError(f"X must have shape {EXPECTED_SHAPE} and dtype float32.")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite tensor values.")
    if y.shape != (EXPECTED_TOTAL,) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"y must be an integer array with shape ({EXPECTED_TOTAL},).")
    if label_id.shape != y.shape or not np.issubdtype(label_id.dtype, np.integer):
        raise ValueError("label_id must be an integer array matching y.")
    if not np.array_equal(y, label_id):
        raise ValueError("y and label_id disagree.")
    if not np.array_equal(np.unique(y), np.arange(len(vocabulary))):
        raise ValueError("Dataset labels must cover exactly the approved vocabulary IDs 0-29.")
    if signs.shape != y.shape or not np.array_equal(signs, np.asarray(vocabulary)[y]):
        raise ValueError("Dataset sign labels do not match metadata vocabulary IDs.")
    if split_names.shape != y.shape or participant_ids.shape != y.shape:
        raise ValueError("split and participant_id arrays must match the sample count.")
    if not np.issubdtype(participant_ids.dtype, np.integer):
        raise ValueError("participant_id must contain integer signer IDs.")

    actual_splits = set(split_names.tolist())
    if actual_splits != set(EXPECTED_SPLIT_COUNTS):
        raise ValueError(f"Dataset splits must be exactly {sorted(EXPECTED_SPLIT_COUNTS)}.")

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, expected_count in EXPECTED_SPLIT_COUNTS.items():
        mask = split_names == split_name
        if int(mask.sum()) != expected_count:
            raise ValueError(
                f"{split_name} split must contain {expected_count} samples; got {int(mask.sum())}."
            )
        class_counts = np.bincount(y[mask], minlength=len(vocabulary))
        if not np.all(class_counts == EXPECTED_SPLIT_LABEL_COUNTS[split_name]):
            raise ValueError(
                f"Each word in {split_name} must contain "
                f"{EXPECTED_SPLIT_LABEL_COUNTS[split_name]} samples."
            )

        actual_signers = set(int(value) for value in np.unique(participant_ids[mask]))
        if actual_signers != expected_signers[split_name]:
            raise ValueError(f"{split_name} signer IDs do not match approved metadata.")
        splits[split_name] = (X[mask], y[mask])

    return WordDataset(splits=splits, vocabulary=vocabulary)
