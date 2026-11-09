"""Check word accuracy gate against threshold and update word_gate.json.

Exits with non-zero code if held-out signer top-1 accuracy is below SIGNBRIDGE_WORD_GATE.
Status values:
  - 'available': top-1 >= gate
  - 'experimental': 0.50 <= top-1 < gate
  - 'withheld': top-1 < 0.50 or no model metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path


def check_gate(
    metrics_path: Path | str = "backend/models/word_metrics.json",
    gate_output_path: Path | str = "backend/models/word_gate.json",
    gate_threshold: float | None = None,
) -> tuple[str, float, float]:
    """Check word model metrics against the configured accuracy gate.

    Args:
        metrics_path: Path to word_metrics.json or fallback to word_eval_metrics.json.
        gate_output_path: Path to write word_gate.json.
        gate_threshold: Accuracy threshold (default from SIGNBRIDGE_WORD_GATE or 0.75).

    Returns:
        Tuple of (status_string, top1_accuracy, gate_threshold).
    """
    if gate_threshold is None:
        gate_env = os.environ.get("SIGNBRIDGE_WORD_GATE")
        gate_threshold = float(gate_env) if gate_env else 0.75

    m_path = Path(metrics_path)
    # Check alternate naming if default does not exist
    if not m_path.exists():
        alt_path = Path("backend/models/word_eval_metrics.json")
        if alt_path.exists():
            m_path = alt_path

    top1 = 0.0
    status = "withheld"

    if m_path.exists():
        try:
            with open(m_path, encoding="utf-8") as f:
                data = json.load(f)

            # Support different metric keys: 'top-1', 'top1', 'test_accuracy', 'accuracy'
            top1 = float(
                data.get("top-1")
                or data.get("top1")
                or data.get("test_accuracy")
                or data.get("accuracy")
                or 0.0
            )

            if top1 >= gate_threshold:
                status = "available"
            elif top1 >= 0.50:
                status = "experimental"
            else:
                status = "withheld"
        except Exception as e:
            print(f"Warning: Failed to parse metrics file {m_path}: {e}")
            status = "withheld"

    out_data = {
        "status": status,
        "top1": round(top1, 4),
        "gate": round(gate_threshold, 4),
        "passed": status == "available",
    }

    out_file = Path(gate_output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    return status, top1, gate_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Check word accuracy gate against threshold")
    parser.add_argument("--metrics", default="backend/models/word_metrics.json", help="Path to metrics file")
    parser.add_argument("--output", default="backend/models/word_gate.json", help="Path to write gate status")
    parser.add_argument("--gate", type=float, default=None, help="Gate threshold (default 0.75)")
    args = parser.parse_args()

    status, top1, gate = check_gate(
        metrics_path=args.metrics,
        gate_output_path=args.output,
        gate_threshold=args.gate,
    )

    print(f"Word accuracy gate check: status='{status}', top-1={top1:.4f}, gate={gate:.4f}")

    if status != "available":
        print(f"ERROR: Model accuracy {top1:.4f} did not meet required gate {gate:.4f}")
        sys.exit(1)
    else:
        print("SUCCESS: Model accuracy passed the required gate.")
        sys.exit(0)


if __name__ == "__main__":
    main()
