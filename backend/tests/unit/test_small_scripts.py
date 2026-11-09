"""Unit tests for small utility scripts: seed_dictionary, build_templates, and check_gate."""

import json
from pathlib import Path

from backend.scripts.build_templates import build_templates
from backend.scripts.check_gate import check_gate
from backend.scripts.seed_dictionary import seed_dictionary


def test_seed_dictionary_creates_valid_wordlist(tmp_path: Path) -> None:
    """Verify seed_dictionary creates >= 2000 lowercase, alphabetic, deduplicated words."""
    words_file = tmp_path / "words.txt"
    count = seed_dictionary(path=words_file, min_words=2000)

    assert count >= 2000
    assert words_file.is_file()

    with open(words_file, encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    assert len(words) == count
    # Deduplicated check
    assert len(words) == len(set(words))
    # All lowercase alphabetic
    for w in words:
        assert w.islower()
        assert w.isalpha()


def test_build_templates_fallback_and_csv(tmp_path: Path) -> None:
    """Verify build_templates produces 21x3 coordinates for all 26 letters."""
    out_file = tmp_path / "templates.json"
    templates = build_templates(dataset_path=tmp_path / "missing.csv", output_path=out_file)

    assert len(templates) == 26
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        assert letter in templates
        pts = templates[letter]
        assert len(pts) == 21
        for pt in pts:
            assert len(pt) == 3


def test_check_gate_statuses(tmp_path: Path) -> None:
    """Verify check_gate computes correct status according to top-1 accuracy."""
    gate_out = tmp_path / "word_gate.json"

    # 1. Available: top1 >= gate (e.g. 0.85 >= 0.75)
    metrics_file = tmp_path / "metrics_avail.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({"top1": 0.85}, f)

    status, top1, gate = check_gate(metrics_file, gate_out, gate_threshold=0.75)
    assert status == "available"
    assert top1 == 0.85
    assert gate == 0.75

    # 2. Experimental: 0.50 <= top1 < 0.75
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({"top1": 0.65}, f)

    status, top1, gate = check_gate(metrics_file, gate_out, gate_threshold=0.75)
    assert status == "experimental"

    # 3. Withheld: top1 < 0.50
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({"top1": 0.42}, f)

    status, top1, gate = check_gate(metrics_file, gate_out, gate_threshold=0.75)
    assert status == "withheld"
