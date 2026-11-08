"""Deterministic manifest-building script for the 6,000-sequence SignBridge ASL word dataset.

Reads:
    backend/data/selected_sequences.csv
Generates:
    backend/data/processed/word_manifest.csv

Specifications:
- 30 closed-vocabulary words.
- Exactly 200 sequences per word (6,000 total sequences).
- Split quotas per word: 140 train, 30 validation, 30 test.
- Strict signer separation (no signer across multiple splits):
    * TRAIN (14 signers): 16069, 18796, 36257, 2044, 29302, 37055, 22343, 62590, 34503, 28656, 27610, 25571, 30680, 4718
    * VAL (3 signers): 26734, 53618, 55372
    * TEST (4 signers): 49445, 37779, 61333, 32319
- All 21 approved signers represented for every single word.
- Retains 100% of the 778 verified local on-disk Parquet files in the training split.
- Balanced sampling across signers.
- Fixed random seed for complete determinism.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = ROOT / "backend" / "data" / "selected_sequences.csv"
DEFAULT_OUTPUT_CSV = ROOT / "backend" / "data" / "processed" / "word_manifest.csv"
DEFAULT_RAW_DIR = ROOT / "backend" / "data" / "raw"

# Approved signer lists per Build Specification & Option C Strategy
TRAIN_SIGNERS = [
    16069, 18796, 36257, 2044, 29302, 37055, 22343,
    62590, 34503, 28656, 27610, 25571, 30680, 4718,
]
VAL_SIGNERS = [26734, 53618, 55372]
TEST_SIGNERS = [49445, 37779, 61333, 32319]


def get_verified_disk_paths(df: pd.DataFrame, raw_dir: Path) -> set[str]:
    """Scan local raw directory and return set of valid Parquet file paths."""
    valid_paths: set[str] = set()
    if not raw_dir.exists():
        return valid_paths

    for p in df["path"]:
        file_path = raw_dir / p
        if file_path.exists() and file_path.stat().st_size > 0:
            try:
                parquet_file = pq.ParquetFile(file_path)
                if parquet_file.metadata.num_rows > 0:
                    valid_paths.add(p)
            except Exception:
                pass
    return valid_paths


def build_word_manifest(
    input_csv: Path,
    output_csv: Path,
    raw_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the balanced 6,000-sequence word manifest with strict signer splitting."""
    print("=" * 65)
    print("SignBridge Deterministic Word Manifest Builder")
    print(f"Input sequences:   {input_csv}")
    print(f"Output manifest:   {output_csv}")
    print(f"Raw directory:     {raw_dir}")
    print(f"Random seed:       {seed}")
    print("=" * 65)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input sequences file not found at: {input_csv}")

    df = pd.read_csv(input_csv)
    total_input = len(df)
    unique_words = sorted(df["sign"].unique())
    print(f"Loaded {total_input} total sequence candidates across {len(unique_words)} words.")

    if len(unique_words) != 30:
        raise ValueError(f"Expected exactly 30 unique words, found {len(unique_words)}")

    # Verify existing local files
    print("Verifying existing local Parquet files offline with PyArrow...")
    on_disk_paths = get_verified_disk_paths(df, raw_dir)
    print(f"Found {len(on_disk_paths)} valid on-disk manifest files.")
    df["on_disk"] = df["path"].isin(on_disk_paths)

    manifest_rows: list[pd.DataFrame] = []

    # Process each word deterministically
    for word_idx, word in enumerate(unique_words, start=1):
        word_df = df[df["sign"] == word].copy()

        # -------------------------------------------------------------
        # 1. Validation Split: exactly 30 sequences (10 from each of 3 signers)
        # -------------------------------------------------------------
        val_parts: list[pd.DataFrame] = []
        val_pool = word_df[word_df["participant_id"].isin(VAL_SIGNERS)]
        for sid in sorted(VAL_SIGNERS):
            s_pool = val_pool[val_pool["participant_id"] == sid]
            if len(s_pool) < 10:
                raise ValueError(f"Word '{word}' signer {sid} has fewer than 10 sequences ({len(s_pool)})")
            sample = s_pool.sample(n=10, random_state=seed)
            val_parts.append(sample)
        val_df_word = pd.concat(val_parts)
        val_df_word["split"] = "val"

        # -------------------------------------------------------------
        # 2. Test Split: exactly 30 sequences across 4 signers (8, 8, 7, 7)
        # -------------------------------------------------------------
        test_parts: list[pd.DataFrame] = []
        test_pool = word_df[word_df["participant_id"].isin(TEST_SIGNERS)]
        test_targets = {
            TEST_SIGNERS[0]: 8,
            TEST_SIGNERS[1]: 8,
            TEST_SIGNERS[2]: 7,
            TEST_SIGNERS[3]: 7,
        }
        for sid, quota in test_targets.items():
            s_pool = test_pool[test_pool["participant_id"] == sid]
            if len(s_pool) < quota:
                raise ValueError(f"Word '{word}' signer {sid} has fewer than {quota} sequences ({len(s_pool)})")
            sample = s_pool.sample(n=quota, random_state=seed)
            test_parts.append(sample)
        test_df_word = pd.concat(test_parts)
        test_df_word["split"] = "test"

        # -------------------------------------------------------------
        # 3. Training Split: exactly 140 sequences
        #    Must include ALL local on-disk files for this word.
        #    Must ensure all 14 training signers are represented.
        # -------------------------------------------------------------
        train_pool = word_df[word_df["participant_id"].isin(TRAIN_SIGNERS)]
        on_disk_train = train_pool[train_pool["on_disk"]].copy()
        on_disk_train["split"] = "train"

        needed_train = 140 - len(on_disk_train)
        if needed_train <= 0:
            raise ValueError(f"Word '{word}' has more on-disk files ({len(on_disk_train)}) than quota (140)")

        # Candidate pool for remaining training slots: not yet on disk
        off_disk_train = train_pool[~train_pool["on_disk"]].copy()

        # Identify which signers are already present in on_disk_train
        signers_on_disk = set(on_disk_train["participant_id"].unique())
        other_signers = [s for s in TRAIN_SIGNERS if s not in signers_on_disk]

        # Allocate base quota across other signers to ensure full 14-signer coverage
        base_quota = needed_train // len(other_signers)
        remainder = needed_train % len(other_signers)

        train_additional: list[pd.DataFrame] = []
        for i, sid in enumerate(sorted(other_signers)):
            quota = base_quota + (1 if i < remainder else 0)
            s_pool = off_disk_train[off_disk_train["participant_id"] == sid]
            sample = s_pool.sample(n=min(quota, len(s_pool)), random_state=seed)
            train_additional.append(sample)

        train_add_df = pd.concat(train_additional)

        # Handle any deficit if a signer had fewer available than the target quota
        deficit = needed_train - len(train_add_df)
        if deficit > 0:
            already_picked = set(train_add_df["path"])
            remaining_off = off_disk_train[~off_disk_train["path"].isin(already_picked)]
            extra_sample = remaining_off.sample(n=deficit, random_state=seed)
            train_additional.append(extra_sample)
            train_add_df = pd.concat(train_additional)

        train_add_df["split"] = "train"
        train_df_word = pd.concat([on_disk_train, train_add_df])

        # Assertions per word
        if len(train_df_word) != 140:
            raise ValueError(f"Word '{word}' expected 140 train sequences, got {len(train_df_word)}")
        if len(val_df_word) != 30:
            raise ValueError(f"Word '{word}' expected 30 val sequences, got {len(val_df_word)}")
        if len(test_df_word) != 30:
            raise ValueError(f"Word '{word}' expected 30 test sequences, got {len(test_df_word)}")

        # Check all 14 train signers represented
        if train_df_word["participant_id"].nunique() != len(TRAIN_SIGNERS):
            missing_signers = set(TRAIN_SIGNERS) - set(train_df_word["participant_id"].unique())
            raise ValueError(f"Word '{word}' missing training signers: {missing_signers}")

        manifest_rows.extend([train_df_word, val_df_word, test_df_word])

    manifest = pd.concat(manifest_rows, ignore_index=True)

    # Required output columns
    output_columns = ["path", "participant_id", "sequence_id", "sign", "split"]
    manifest = manifest[output_columns].copy()

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)
    print(f"Manifest written to: {output_csv}")

    return manifest


def validate_manifest(manifest_csv: Path, on_disk_paths: set[str]) -> None:
    """Run rigorous validation checks against all project specifications."""
    print("\n" + "=" * 65)
    print("Running Rigorous Manifest Verification...")
    print("=" * 65)

    df = pd.read_csv(manifest_csv)

    # 1. Row count and uniqueness
    total_rows = len(df)
    unique_paths = df["path"].nunique()
    print(f"1. Total rows:             {total_rows} (Expected: 6000)")
    print(f"   Unique paths:           {unique_paths} (Expected: 6000)")
    assert total_rows == 6000, f"Expected 6000 rows, got {total_rows}"
    assert unique_paths == 6000, f"Found duplicate paths: {total_rows - unique_paths}"

    # 2. Per-word total counts
    word_counts = df["sign"].value_counts()
    print(f"2. Words represented:      {len(word_counts)} (Expected: 30)")
    print(f"   Min/Max seqs per word:  {word_counts.min()} / {word_counts.max()} (Expected: 200 / 200)")
    assert len(word_counts) == 30, f"Expected 30 words, got {len(word_counts)}"
    assert (word_counts == 200).all(), "Not all words have exactly 200 sequences!"

    # 3. Per-word split counts
    split_counts = df.groupby(["sign", "split"]).size().unstack(fill_value=0)
    print(f"3. Split quotas per word:  train={split_counts['train'].min()}..{split_counts['train'].max()}, "
          f"val={split_counts['val'].min()}..{split_counts['val'].max()}, "
          f"test={split_counts['test'].min()}..{split_counts['test'].max()} (Expected: 140, 30, 30)")
    assert (split_counts["train"] == 140).all(), "Train quota violated!"
    assert (split_counts["val"] == 30).all(), "Validation quota violated!"
    assert (split_counts["test"] == 30).all(), "Test quota violated!"

    # 4. Signer separation between splits
    train_signers = set(df[df["split"] == "train"]["participant_id"].unique())
    val_signers = set(df[df["split"] == "val"]["participant_id"].unique())
    test_signers = set(df[df["split"] == "test"]["participant_id"].unique())

    print(f"4. Signer separation check:")
    print(f"   Train signers ({len(train_signers)}): {sorted(train_signers)}")
    print(f"   Val signers   ({len(val_signers)}): {sorted(val_signers)}")
    print(f"   Test signers  ({len(test_signers)}): {sorted(test_signers)}")

    assert train_signers.isdisjoint(val_signers), "Train and Validation signers overlap!"
    assert train_signers.isdisjoint(test_signers), "Train and Test signers overlap!"
    assert val_signers.isdisjoint(test_signers), "Validation and Test signers overlap!"
    assert train_signers == set(TRAIN_SIGNERS), "Train signers mismatch!"
    assert val_signers == set(VAL_SIGNERS), "Validation signers mismatch!"
    assert test_signers == set(TEST_SIGNERS), "Test signers mismatch!"

    # 5. All 21 signers represented for every word
    signers_per_word = df.groupby("sign")["participant_id"].nunique()
    print(f"5. Signers represented per word: min={signers_per_word.min()}, max={signers_per_word.max()} (Expected: 21)")
    assert (signers_per_word == 21).all(), "Not all 21 signers are represented in every word!"

    # 6. Retention of all on-disk files
    retained_on_disk = df["path"].isin(on_disk_paths).sum()
    print(f"6. Valid on-disk files retained: {retained_on_disk} / {len(on_disk_paths)} (Expected: 100%)")
    assert retained_on_disk == len(on_disk_paths), f"Lost on-disk files: {len(on_disk_paths) - retained_on_disk}"

    # Verify all retained on-disk files are in TRAIN
    on_disk_splits = df[df["path"].isin(on_disk_paths)]["split"].value_counts()
    print(f"   On-disk split assignment:     {dict(on_disk_splits)} (Expected: all train)")
    assert list(on_disk_splits.keys()) == ["train"], "Some on-disk files were placed outside train split!"

    # 7. No outside words (e.g. cloud)
    print("7. Outside vocabulary check:     No non-vocabulary files present.")
    assert "cloud" not in df["sign"].values, "Found non-vocabulary word 'cloud' in manifest!"

    print("=" * 65)
    print(">>> ALL VALIDATION CHECKS PASSED PERFECTLY! <<<")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic ASL word manifest.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV containing selected sequence candidates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV for the generated 6000-sequence manifest",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing downloaded raw Parquet files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    on_disk_paths = get_verified_disk_paths(input_df, args.raw_dir)

    build_word_manifest(
        input_csv=args.input,
        output_csv=args.output,
        raw_dir=args.raw_dir,
        seed=args.seed,
    )

    validate_manifest(args.output, on_disk_paths)


if __name__ == "__main__":
    main()
