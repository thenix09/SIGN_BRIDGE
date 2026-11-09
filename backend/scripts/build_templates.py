"""Build normalized mean landmark templates per letter and save to templates.json."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

# Default canonical hand geometry (21 points) for fingerspelling fallback
DEFAULT_HAND_POINTS: list[list[float]] = [
    [0.0, 0.0, 0.0],       # 0: Wrist
    [-0.05, -0.1, 0.0],    # 1: Thumb CMC
    [-0.09, -0.18, 0.0],   # 2: Thumb MCP
    [-0.12, -0.24, 0.0],   # 3: Thumb IP
    [-0.14, -0.30, 0.0],   # 4: Thumb TIP
    [-0.05, -0.25, 0.0],   # 5: Index MCP
    [-0.06, -0.38, 0.0],   # 6: Index PIP
    [-0.06, -0.48, 0.0],   # 7: Index DIP
    [-0.06, -0.56, 0.0],   # 8: Index TIP
    [0.0, -0.26, 0.0],     # 9: Middle MCP
    [0.0, -0.40, 0.0],     # 10: Middle PIP
    [0.0, -0.51, 0.0],     # 11: Middle DIP
    [0.0, -0.60, 0.0],     # 12: Middle TIP
    [0.05, -0.24, 0.0],    # 13: Ring MCP
    [0.05, -0.37, 0.0],    # 14: Ring PIP
    [0.05, -0.47, 0.0],    # 15: Ring DIP
    [0.05, -0.55, 0.0],    # 16: Ring TIP
    [0.09, -0.20, 0.0],    # 17: Pinky MCP
    [0.11, -0.30, 0.0],    # 18: Pinky PIP
    [0.12, -0.38, 0.0],    # 19: Pinky DIP
    [0.13, -0.45, 0.0],    # 20: Pinky TIP
]


def generate_fallback_templates() -> dict[str, list[list[float]]]:
    """Generate default 21x3 normalized templates for letters A-Z."""
    letters = [chr(ord("A") + i) for i in range(26)]
    templates: dict[str, list[list[float]]] = {}
    for char in letters:
        # Clone default geometry
        pts = [[round(coord, 4) for coord in p] for p in DEFAULT_HAND_POINTS]
        templates[char] = pts
    return templates


def build_templates(
    dataset_path: Path | str = "backend/data/samples.csv",
    output_path: Path | str = "backend/models/templates.json",
) -> dict[str, Any]:
    """Compute mean normalized vector per letter from dataset or generate defaults.

    Args:
        dataset_path: Path to samples CSV (header: label,group,f0..f62).
        output_path: Path to save templates.json.

    Returns:
        Dict mapping uppercase letter ('A'..'Z') to 21x3 float coordinates.
    """
    csv_file = Path(dataset_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    templates: dict[str, list[list[float]]] = {}

    if csv_file.exists():
        grouped_features: dict[str, list[list[float]]] = {}
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if len(row) >= 65:  # label, group, f0..f62
                    label = row[0].strip().upper()
                    features = [float(val) for val in row[2:65]]
                    grouped_features.setdefault(label, []).append(features)

        for label, rows in grouped_features.items():
            if rows:
                n = len(rows)
                mean_feats = [sum(r[i] for r in rows) / n for i in range(63)]
                # Reshape 63 floats to 21x3
                reshaped = [
                    [round(mean_feats[i * 3], 4), round(mean_feats[i * 3 + 1], 4), round(mean_feats[i * 3 + 2], 4)]
                    for i in range(21)
                ]
                templates[label] = reshaped

    # Fill any missing letters with fallback defaults
    fallback = generate_fallback_templates()
    for letter, pts in fallback.items():
        if letter not in templates:
            templates[letter] = pts

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2)

    return templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mean landmark templates per letter")
    parser.add_argument("--dataset", default="backend/data/samples.csv", help="Path to samples CSV")
    parser.add_argument("--output", default="backend/models/templates.json", help="Path to templates.json")
    args = parser.parse_args()

    res = build_templates(dataset_path=args.dataset, output_path=args.output)
    print(f"Successfully generated templates for {len(res)} letters in {args.output}")


if __name__ == "__main__":
    main()
