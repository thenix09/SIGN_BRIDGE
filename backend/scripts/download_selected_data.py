"""Download selected Parquet landmark files from Kaggle asl-signs competition.

Features:
- Configurable concurrency with a shared global request rate limiter.
- Independent thread-local KaggleApi clients to avoid connection pool contention.
- Resumable: validates existing files with PyArrow before skipping.
- Isolated staging preserves existing destination files and archives.
- Coordinated 429 backoff: on HTTP 429, pauses all workers globally for >= 60 seconds (respecting Retry-After).
- Live disk space and failure rate monitoring.
- Never prints or exposes Kaggle credentials.
"""

import argparse
import math
import os
import random
import shutil
import tempfile
import threading
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal, Protocol

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "backend" / "data" / "selected_parquet_paths.txt"
RAW_DIR = ROOT / "backend" / "data" / "raw"

# Thread-local storage for independent KaggleApi instances
_thread_local = threading.local()


class CompetitionFileApi(Protocol):
    def competition_download_file(
        self,
        competition: str,
        file_name: str,
        path: str,
        quiet: bool,
        unzip: bool,
        force: bool,
    ) -> None: ...


class GlobalRequestRateLimiter:
    """Thread-safe global rate limiter enforcing an aggregate requests-per-minute cap
    across all concurrent workers, plus coordinated pauses on HTTP 429.
    """

    def __init__(self, requests_per_minute: float = 100.0) -> None:
        self._lock = threading.Lock()
        self._requests_per_minute = max(requests_per_minute, 1.0)
        self._interval = 60.0 / self._requests_per_minute  # e.g. 0.6s at 100 rpm
        self._next_request_time = 0.0
        self._pause_until = 0.0
        self._429_count = 0
        self._retry_count = 0

    @property
    def event_count_429(self) -> int:
        with self._lock:
            return self._429_count

    @property
    def retry_count(self) -> int:
        with self._lock:
            return self._retry_count

    @property
    def cooldown_remaining_seconds(self) -> float:
        with self._lock:
            return max(self._pause_until - time.monotonic(), 0.0)

    def record_retry(self) -> None:
        with self._lock:
            self._retry_count += 1

    def acquire(self, stop_event: threading.Event | None = None) -> bool:
        """Block until it is safe to issue the next request according to the global rate cap
        and any active 429 cooldown.
        """
        while True:
            with self._lock:
                now = time.monotonic()
                next_request_time = max(self._pause_until, self._next_request_time)
                wait_time = next_request_time - now
                if wait_time <= 0:
                    self._next_request_time = now + self._interval
                    return True

            if stop_event is not None:
                if stop_event.wait(wait_time):
                    return False
            else:
                time.sleep(wait_time)

    def report_429(self, retry_after: float | None = None, min_pause: float = 60.0) -> None:
        """Coordinated pause when HTTP 429 is encountered. Respects Retry-After or min 60s."""
        pause_seconds = max(retry_after if retry_after is not None else 0.0, min_pause)
        with self._lock:
            self._429_count += 1
            now = time.monotonic()
            target_time = now + pause_seconds
            if target_time > self._pause_until:
                self._pause_until = target_time
                resume_time = time.strftime(
                    "%H:%M:%S",
                    time.localtime(time.time() + pause_seconds),
                )
                print(
                    f"\n[GLOBAL RATE LIMIT 429] (Event #{self._429_count}) Pausing all workers for {pause_seconds:.1f}s "
                    f"until {resume_time}...",
                    flush=True,
                )


rate_limiter = GlobalRequestRateLimiter(requests_per_minute=100.0)


def extract_retry_after(error: Exception) -> float | None:
    """Extract Retry-After as seconds from either supported HTTP header format."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    retry_after = headers.get("Retry-After") if headers is not None else None
    if not retry_after:
        return None
    try:
        seconds = float(retry_after)
        return max(seconds, 0.0) if math.isfinite(seconds) else None
    except (TypeError, ValueError):
        try:
            retry_time = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=UTC)
        return max((retry_time - datetime.now(UTC)).total_seconds(), 0.0)


def is_http_429(error: Exception) -> bool:
    """Check if exception corresponds to HTTP 429 Too Many Requests."""
    err_str = str(error).lower()
    if "429" in err_str or "too many requests" in err_str or "rate limit" in err_str:
        return True
    if hasattr(error, "response") and getattr(error.response, "status_code", None) == 429:
        return True
    return False


def get_thread_kaggle_api() -> CompetitionFileApi:
    """Obtain or initialize an authenticated KaggleApi client unique to current thread."""
    if not hasattr(_thread_local, "api"):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ModuleNotFoundError as exc:
            if exc.name != "kaggle":
                raise
            raise RuntimeError(
                "Kaggle is required to download dataset files. "
                "Install the project's development requirements."
            ) from exc

        api = KaggleApi()
        api.authenticate()
        _thread_local.api = api
    return _thread_local.api


def is_valid_parquet(file_path: Path) -> bool:
    """Check if file exists and has valid Parquet metadata and rows."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    try:
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows > 0
    except Exception:
        return False


def clean_residual_zips(raw_dir: Path) -> int:
    """Explicitly remove leftover .zip files whose Parquet counterpart is valid."""
    cleaned = 0
    for zip_path in raw_dir.glob("**/*.parquet.zip"):
        parquet_path = zip_path.with_name(zip_path.name[: -len(".zip")])
        if is_valid_parquet(parquet_path):
            zip_path.unlink(missing_ok=True)
            cleaned += 1
    return cleaned


def download_one(
    api: CompetitionFileApi,
    remote_path: str,
    raw_dir: Path,
    max_retries: int = 3,
    stop_event: threading.Event | None = None,
    max_429_events: int = 1,
    initial_429_count: int | None = None,
    min_429_cooldown: float = 120.0,
) -> tuple[Literal["downloaded", "skipped", "failed", "cancelled"], str | None]:
    """Download and extract a single Parquet file if not already valid.

    Args:
        api: Authenticated KaggleApi instance for current thread.
        remote_path: Relative path in Kaggle dataset (e.g. train_landmark_files/16069/1005223850.parquet).
        raw_dir: Local destination root directory.
        max_retries: Maximum number of attempts.
        stop_event: Shared signal to stop queued work after a safety threshold.
        max_429_events: Number of new 429 events allowed before stopping.
        min_429_cooldown: Minimum coordinated pause after HTTP 429, in seconds.

    Returns:
        Tuple of (status, error_message).
    """
    target = raw_dir / remote_path

    if is_valid_parquet(target):
        return "skipped", None

    if target.exists():
        return "failed", f"Refusing to overwrite existing invalid file: {target}"

    target.parent.mkdir(parents=True, exist_ok=True)

    if stop_event is None:
        stop_event = threading.Event()
    if initial_429_count is None:
        initial_429_count = rate_limiter.event_count_429
    last_error: str | None = None

    for attempt in range(1, max_retries + 1):
        if stop_event.is_set():
            return "cancelled", None
        try:
            if not rate_limiter.acquire(stop_event):
                return "cancelled", None

            with tempfile.TemporaryDirectory(prefix="signbridge-download-", dir=raw_dir) as temp_dir:
                staging_dir = Path(temp_dir)
                if stop_event.is_set():
                    return "cancelled", None

                api.competition_download_file(
                    competition="asl-signs",
                    file_name=remote_path,
                    path=str(staging_dir),
                    quiet=True,
                    unzip=True,
                    force=True,
                )

                staged_target = staging_dir / remote_path
                staged_zip = staging_dir / (remote_path + ".zip")
                if not staged_target.exists() and staged_zip.exists():
                    with zipfile.ZipFile(staged_zip, "r") as archive:
                        parquet_members = [
                            member for member in archive.namelist() if member.lower().endswith(".parquet")
                        ]
                        if not parquet_members:
                            raise ValueError(f"No .parquet file found in archive {staged_zip.name}")
                        staged_target.parent.mkdir(parents=True, exist_ok=True)
                        staged_temp = staged_target.with_suffix(".tmp")
                        with archive.open(parquet_members[0]) as source, staged_temp.open("wb") as dest:
                            shutil.copyfileobj(source, dest)
                        staged_temp.replace(staged_target)

                if not is_valid_parquet(staged_target):
                    raise ValueError(f"Downloaded file failed Parquet validation: {staged_target}")

                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if is_valid_parquet(target):
                        return "skipped", None
                    return "failed", f"Refusing to overwrite existing invalid file: {target}"

                try:
                    os.link(staged_target, target)
                except (OSError, NotImplementedError):
                    shutil.copyfile(staged_target, target)
                return "downloaded", None

        except Exception as error:
            last_error = str(error)

            if is_http_429(error):
                retry_after = extract_retry_after(error)
                rate_limiter.report_429(
                    retry_after=retry_after,
                    min_pause=min_429_cooldown,
                )
                if rate_limiter.event_count_429 - initial_429_count >= max_429_events:
                    stop_event.set()
                    return "failed", last_error
            else:
                backoff_time = min(2.0**attempt + random.uniform(0.5, 1.5), 300.0)

            if attempt < max_retries and not is_http_429(error):
                rate_limiter.record_retry()
                time.sleep(backoff_time)

    return "failed", last_error


def worker_task(
    remote_path: str,
    raw_dir: Path,
    stop_event: threading.Event,
    max_429_events: int,
    initial_429_count: int,
    min_429_cooldown: float,
) -> tuple[str, str, str | None]:
    """Worker entry point: obtains thread-local API and downloads file."""
    if stop_event.is_set():
        return remote_path, "cancelled", None

    try:
        if stop_event.is_set():
            return remote_path, "cancelled", None
        api = get_thread_kaggle_api()
    except Exception as error:
        if is_http_429(error):
            rate_limiter.report_429(
                retry_after=extract_retry_after(error),
                min_pause=min_429_cooldown,
            )
            if rate_limiter.event_count_429 - initial_429_count >= max_429_events:
                stop_event.set()
        return remote_path, "failed", str(error)

    status, error_msg = download_one(
        api,
        remote_path,
        raw_dir,
        stop_event=stop_event,
        max_429_events=max_429_events,
        initial_429_count=initial_429_count,
        min_429_cooldown=min_429_cooldown,
    )
    return remote_path, status, error_msg


def run_downloader(
    paths: list[str],
    raw_dir: Path,
    workers: int = 5,
    progress_interval: int = 100,
    max_consecutive_failures: int = 20,
    max_429_events: int = 5,
    min_free_bytes: int = 2 * 1024 * 1024 * 1024,
    min_429_cooldown: float = 120.0,
) -> tuple[int, int, int]:
    """Execute bounded concurrent download across specified paths."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if max_429_events < 1:
        raise ValueError("max_429_events must be at least 1")
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1")
    if min_429_cooldown < 0:
        raise ValueError("min_429_cooldown cannot be negative")

    print("=" * 65)
    print("SignBridge Selected Parquet Downloader (Rate-Controlled)")
    print(f"Total paths to process: {len(paths)}")
    print(f"Worker threads:         {workers}")
    print(f"Global rate limit:      {rate_limiter._requests_per_minute:.0f} req/min (1 req every {rate_limiter._interval:.2f}s)")
    print(f"Progress interval:      every {progress_interval} files")
    print(f"Raw storage directory:  {raw_dir}")
    print("=" * 65, flush=True)

    downloaded = 0
    skipped = 0
    failed = 0
    cancelled = 0
    consecutive_failures = 0
    last_checkpoint_count = 0
    start_time = time.time()
    initial_429_count = rate_limiter.event_count_429
    initial_retry_count = rate_limiter.retry_count
    stop_event = threading.Event()

    max_in_flight = workers
    path_iter = iter(paths)
    in_flight: dict[Future, str] = {}
    total_to_process = len(paths)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Pre-fill worker queue up to bounded limit
        for _ in range(min(max_in_flight, total_to_process)):
            try:
                p = next(path_iter)
                fut = executor.submit(
                    worker_task,
                    p,
                    raw_dir,
                    stop_event,
                    max_429_events,
                    initial_429_count,
                    min_429_cooldown,
                )
                in_flight[fut] = p
            except StopIteration:
                break

        while in_flight:
            # Wait for at least one future to finish
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)

            for fut in done:
                rpath = in_flight.pop(fut)
                try:
                    _, status, err = fut.result()
                except Exception as ex:
                    status = "failed"
                    err = str(ex)

                if status == "downloaded":
                    downloaded += 1
                    consecutive_failures = 0
                elif status == "skipped":
                    skipped += 1
                    consecutive_failures = 0
                elif status == "cancelled":
                    cancelled += 1
                else:
                    failed += 1
                    consecutive_failures += 1
                    print(f"\n[FAILURE] {rpath}: {err}", flush=True)

                processed = downloaded + skipped + failed + cancelled

                # Safety checks before submitting new work
                if stop_event.is_set():
                    path_iter = iter([])
                elif shutil.disk_usage(raw_dir).free < min_free_bytes:
                    free_bytes = shutil.disk_usage(raw_dir).free
                    print(
                        f"\n[SAFETY PAUSE] Low disk space ({free_bytes / (1024**3):.2f} GB free). "
                        "Halting new requests.",
                        flush=True,
                    )
                    stop_event.set()
                    path_iter = iter([])
                elif consecutive_failures >= max_consecutive_failures:
                    print(
                        f"\n[SAFETY PAUSE] {consecutive_failures} consecutive failures. Halting submission.",
                        flush=True,
                    )
                    stop_event.set()
                    path_iter = iter([])
                else:
                    # Submit next task if available
                    try:
                        next_path = next(path_iter)
                        new_fut = executor.submit(
                            worker_task,
                            next_path,
                            raw_dir,
                            stop_event,
                            max_429_events,
                            initial_429_count,
                            min_429_cooldown,
                        )
                        in_flight[new_fut] = next_path
                    except StopIteration:
                        pass

                # Major milestone checkpoint every 500 newly downloaded files
                if downloaded >= last_checkpoint_count + 500:
                    last_checkpoint_count = (downloaded // 500) * 500
                    elapsed_now = time.time() - start_time
                    cur_net_rate = downloaded / elapsed_now if elapsed_now > 0 else 0
                    print(
                        f"\n>>> [MILESTONE CHECKPOINT] Downloaded: {downloaded} | "
                        f"Skipped: {skipped} | Failed: {failed} | "
                        f"429 Events: {rate_limiter.event_count_429} | "
                        f"Net Rate: {cur_net_rate:.2f} files/s | Elapsed: {elapsed_now/60:.1f}m <<<\n",
                        flush=True,
                    )

                # Periodic progress output
                if processed % progress_interval == 0 or processed == total_to_process:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (total_to_process - processed) / rate if rate > 0 else 0
                    print(
                        f"[{processed}/{total_to_process}] "
                        f"Downloaded: {downloaded} | Skipped: {skipped} | Failed: {failed} "
                        f"({rate:.1f} files/s, ETA: {remaining/60:.1f}m)",
                        flush=True,
                    )

    elapsed_total = time.time() - start_time
    effective_rate = downloaded / elapsed_total if elapsed_total > 0 else 0

    print("\n" + "=" * 65)
    print("Download Summary:")
    print(f"Total processed:     {downloaded + skipped + failed + cancelled}")
    print(f"Downloaded (new):    {downloaded}")
    print(f"Skipped (verified):  {skipped}")
    print(f"Failed:              {failed}")
    print(f"Cancelled:           {cancelled}")
    print(f"429 Events:          {rate_limiter.event_count_429}")
    print(f"Retries:             {rate_limiter.retry_count - initial_retry_count}")
    print(f"Elapsed time:        {elapsed_total:.1f}s")
    print(f"Net download rate:   {effective_rate:.2f} files/s")
    print("=" * 65, flush=True)

    return downloaded, skipped, failed


def verify_limiter(num_workers: int = 5, num_requests: int = 10, rpm: float = 100.0) -> None:
    """Automated verification that GlobalRequestRateLimiter strictly throttles across workers."""
    print(
        f"Verifying GlobalRequestRateLimiter with {num_workers} concurrent threads across {num_requests} requests...",
        flush=True,
    )
    limiter = GlobalRequestRateLimiter(requests_per_minute=rpm)
    expected_interval = 60.0 / rpm
    expected_min_time = (num_requests - 1) * expected_interval

    timestamps: list[float] = []
    ts_lock = threading.Lock()

    def dummy_worker() -> None:
        for _ in range(num_requests // num_workers):
            limiter.acquire()
            now = time.time()
            with ts_lock:
                timestamps.append(now)

    t0 = time.time()
    threads = [threading.Thread(target=dummy_worker) for _ in range(num_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_elapsed = time.time() - t0

    timestamps.sort()
    diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

    print(f"Total requests:        {len(timestamps)}", flush=True)
    print(f"Total elapsed time:    {total_elapsed:.2f}s (minimum expected: {expected_min_time:.2f}s)", flush=True)
    print(f"Target inter-request:  {expected_interval:.3f}s", flush=True)
    print(f"Average inter-request: {sum(diffs) / len(diffs):.3f}s", flush=True)
    print(f"Observed global rate:  {(len(timestamps) / total_elapsed) * 60:.1f} req/min (target cap: <= {rpm:.1f} req/min)", flush=True)

    assert total_elapsed >= expected_min_time * 0.95, f"Limiter ran too fast: {total_elapsed:.2f}s"
    print(">>> VERIFICATION SUCCESS: GlobalRequestRateLimiter strictly enforces aggregate limit across workers! <<<\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected ASL signs Parquet sequences.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads (default: 4)",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=8.0,
        help="Global request rate limit in requests per minute (default: 8.0 = 1 req/7.5s)",
    )
    parser.add_argument(
        "--max-429",
        type=int,
        default=1,
        help="Maximum allowed HTTP 429 events before halting submission (default: 1)",
    )
    parser.add_argument(
        "--min-429-cooldown",
        type=float,
        default=120.0,
        help="Minimum global pause after HTTP 429, in seconds (default: 120)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print progress every N processed files (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of files to process in this run (0 = all in manifest)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset index in manifest to start processing from",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check existing vs missing files without downloading",
    )
    parser.add_argument(
        "--clean-zips",
        action="store_true",
        help="Clean up residual .zip files for valid Parquet files and exit",
    )
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="Run strict PyArrow validation on all manifest files present on disk",
    )
    parser.add_argument(
        "--verify-limiter",
        action="store_true",
        help="Run standalone rate limiter verification test across 5 workers and exit",
    )
    args = parser.parse_args()

    if args.verify_limiter:
        verify_limiter(num_workers=args.workers, num_requests=10, rpm=args.rpm)
        return

    if not MANIFEST.exists():
        print(f"Error: Manifest not found at {MANIFEST}")
        return

    if args.clean_zips:
        cleaned = clean_residual_zips(RAW_DIR)
        print(f"Cleaned {cleaned} residual .zip files.")
        return

    paths = [
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if args.verify_all:
        print(f"Verifying {len(paths)} manifest files on disk with PyArrow...")
        valid = 0
        missing = 0
        corrupt = 0
        for p in paths:
            target = RAW_DIR / p
            if not target.exists():
                missing += 1
            elif is_valid_parquet(target):
                valid += 1
            else:
                corrupt += 1
        print(f"Verification Results: {valid} valid, {missing} missing, {corrupt} corrupt.")
        return

    # Update global rate limiter RPM if specified
    global rate_limiter
    rate_limiter = GlobalRequestRateLimiter(requests_per_minute=args.rpm)

    if args.offset > 0:
        paths = paths[args.offset :]

    if args.limit > 0:
        paths = paths[: args.limit]

    if args.dry_run:
        print("Checking existing files on disk...")
        valid_count = 0
        for p in paths:
            if is_valid_parquet(RAW_DIR / p):
                valid_count += 1
        missing_count = len(paths) - valid_count
        print(f"Status: {valid_count} already valid, {missing_count} missing.")
        return

    run_downloader(
        paths=paths,
        raw_dir=RAW_DIR,
        workers=args.workers,
        progress_interval=args.progress_interval,
        max_429_events=args.max_429,
        min_429_cooldown=args.min_429_cooldown,
    )


if __name__ == "__main__":
    main()
