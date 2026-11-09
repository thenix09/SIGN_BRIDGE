"""Benchmark letter recognition latency and frame throughput."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.events import EventBus  # noqa: E402
from backend.app.recognition.base import FakeRecognizer, SignRecognizer  # noqa: E402
from backend.app.recognition.words import SuggestionEngine  # noqa: E402
from backend.app.services.sessions import SessionManager  # noqa: E402
from backend.app.services.sign_to_text import SignToTextService  # noqa: E402


class BenchFactory:
    """Mock factory returning FakeRecognizer for performance benchmarking."""

    @classmethod
    def create(cls, language: str) -> SignRecognizer:
        return FakeRecognizer(default_label="A", default_confidence=0.95)


def generate_mock_landmarks() -> list[list[float]]:
    """Generate 21 sample landmark coordinates in [x, y, z] format."""
    return [[0.5 + (i * 0.01), 0.5 + (i * 0.01), 0.0] for i in range(21)]


def benchmark_letter_latency(iterations: int = 1000) -> dict[str, Any]:
    """Benchmark end-to-end letter pipeline processing latency over multiple iterations.

    Target from specification:
        Median letter-frame latency under 100 ms locally.
    """
    sessions = SessionManager()
    bus = EventBus()
    suggestions = SuggestionEngine()

    service = SignToTextService(
        factory=BenchFactory,  # type: ignore[arg-type]
        sessions=sessions,
        suggestions=suggestions,
        bus=bus,
    )

    landmarks = generate_mock_landmarks()
    session_id = "bench-session"

    # Warmup (20 frames)
    for _ in range(20):
        service.process_frame(
            session_id=session_id,
            language="ASL",
            raw_landmarks=landmarks,
            handedness="Right",
        )

    latencies_ms: list[float] = []

    for _ in range(iterations):
        start = time.perf_counter()
        service.process_frame(
            session_id=session_id,
            language="ASL",
            raw_landmarks=landmarks,
            handedness="Right",
        )
        duration_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(duration_ms)

    latencies_sorted = sorted(latencies_ms)
    median_val = statistics.median(latencies_sorted)
    mean_val = statistics.mean(latencies_sorted)
    min_val = min(latencies_sorted)
    max_val = max(latencies_sorted)
    p95_val = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99_val = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    target_met = median_val < 100.0

    return {
        "iterations": iterations,
        "min_ms": round(min_val, 3),
        "max_ms": round(max_val, 3),
        "mean_ms": round(mean_val, 3),
        "median_ms": round(median_val, 3),
        "p95_ms": round(p95_val, 3),
        "p99_ms": round(p99_val, 3),
        "target_ms": 100.0,
        "target_met": target_met,
        "estimated_fps": round(1000.0 / mean_val, 1) if mean_val > 0 else 0.0,
    }


def main() -> None:
    print("=" * 60)
    print("SignBridge Performance Benchmark: Letter Recognition Pipeline")
    print("=" * 60)
    results = benchmark_letter_latency(iterations=1000)

    print(f"Iterations:        {results['iterations']}")
    print(f"Min Latency:       {results['min_ms']} ms")
    print(f"Mean Latency:      {results['mean_ms']} ms")
    print(f"Median (p50):      {results['median_ms']} ms")
    print(f"Max Latency:       {results['max_ms']} ms")
    print(f"95th Percentile:   {results['p95_ms']} ms")
    print(f"99th Percentile:   {results['p99_ms']} ms")
    print(f"Target (<100ms):   {results['median_ms']} ms <= 100.0 ms -> {'PASS' if results['target_met'] else 'FAIL'}")
    print(f"Throughput:        ~{results['estimated_fps']} frames/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
