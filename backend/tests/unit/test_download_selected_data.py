from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from backend.scripts import download_selected_data as downloader


class FakeKaggleApi:
    def __init__(self) -> None:
        self.calls = 0

    def competition_download_file(
        self,
        competition: str,
        file_name: str,
        path: str,
        quiet: bool,
        unzip: bool,
        force: bool,
    ) -> None:
        self.calls += 1
        assert competition == "asl-signs"
        assert quiet and unzip and force
        destination = Path(path) / file_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"value": [1]}), destination)


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"value": [1]}), path)


def test_valid_existing_file_and_sidecars_are_preserved(tmp_path: Path) -> None:
    remote_path = "train_landmark_files/123/valid.parquet"
    target = tmp_path / remote_path
    _write_parquet(target)
    archive = tmp_path / f"{remote_path}.zip"
    archive.write_bytes(b"keep archive")
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(b"keep temporary file")
    api = FakeKaggleApi()

    result = downloader.download_one(api, remote_path, tmp_path)

    assert result == ("skipped", None)
    assert api.calls == 0
    assert archive.read_bytes() == b"keep archive"
    assert temporary.read_bytes() == b"keep temporary file"


def test_invalid_existing_file_is_not_deleted_or_overwritten(tmp_path: Path) -> None:
    remote_path = "train_landmark_files/123/corrupt.parquet"
    target = tmp_path / remote_path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"preserve this file")
    api = FakeKaggleApi()

    status, error = downloader.download_one(api, remote_path, tmp_path)

    assert status == "failed"
    assert error is not None and "Refusing to overwrite" in error
    assert target.read_bytes() == b"preserve this file"
    assert api.calls == 0


def test_download_stages_output_and_preserves_existing_archive(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    remote_path = "train_landmark_files/123/new.parquet"
    archive = tmp_path / f"{remote_path}.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"pre-existing archive")
    api = FakeKaggleApi()
    monkeypatch.setattr(
        downloader,
        "rate_limiter",
        downloader.GlobalRequestRateLimiter(requests_per_minute=60_000),
    )

    result = downloader.download_one(api, remote_path, tmp_path)

    assert result == ("downloaded", None)
    assert downloader.is_valid_parquet(tmp_path / remote_path)
    assert archive.read_bytes() == b"pre-existing archive"


def test_rate_limiter_rechecks_pause_after_waiting(monkeypatch: Any) -> None:
    now = [100.0]
    pauses: list[float] = []
    limiter = downloader.GlobalRequestRateLimiter(requests_per_minute=60.0)
    limiter._next_request_time = 101.0

    def sleep(seconds: float) -> None:
        pauses.append(seconds)
        if len(pauses) == 1:
            limiter.report_429(retry_after=10.0, min_pause=10.0)
        now[0] += seconds

    monkeypatch.setattr(downloader.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(downloader.time, "sleep", sleep)

    assert limiter.acquire()
    assert now[0] >= 110.0
    assert len(pauses) == 2
    assert pauses[1] >= 9.0


def test_retry_after_accepts_http_date() -> None:
    retry_time = datetime.now(UTC) + timedelta(seconds=30)
    error = Exception("rate limited")
    response = SimpleNamespace(headers={"Retry-After": format_datetime(retry_time, usegmt=True)})
    setattr(error, "response", response)

    seconds = downloader.extract_retry_after(error)

    assert seconds is not None and 0 < seconds <= 30


def test_429_threshold_stops_retries_without_touching_existing_data(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    class RateLimitError(Exception):
        def __init__(self) -> None:
            super().__init__("429")
            self.response = SimpleNamespace(status_code=429, headers={"Retry-After": "60"})

    class RateLimitedApi:
        calls = 0

        def competition_download_file(
            self,
            competition: str,
            file_name: str,
            path: str,
            quiet: bool,
            unzip: bool,
            force: bool,
        ) -> None:
            self.calls += 1
            raise RateLimitError()

    api = RateLimitedApi()
    stop_event = downloader.threading.Event()
    monkeypatch.setattr(
        downloader,
        "rate_limiter",
        downloader.GlobalRequestRateLimiter(requests_per_minute=60_000),
    )

    result = downloader.download_one(
        api,
        "train_landmark_files/123/new.parquet",
        tmp_path,
        stop_event=stop_event,
        max_429_events=1,
        initial_429_count=0,
    )

    assert result[0] == "failed"
    assert api.calls == 1
    assert stop_event.is_set()
    assert downloader.rate_limiter.cooldown_remaining_seconds >= 119


def test_transient_failures_use_exponential_backoff_and_count_retries(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    class TransientApi:
        def __init__(self) -> None:
            self.calls = 0

        def competition_download_file(
            self,
            competition: str,
            file_name: str,
            path: str,
            quiet: bool,
            unzip: bool,
            force: bool,
        ) -> None:
            self.calls += 1
            if self.calls < 3:
                raise OSError("temporary network failure")
            destination = Path(path) / file_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({"value": [1]}), destination)

    api = TransientApi()
    delays: list[float] = []
    limiter = downloader.GlobalRequestRateLimiter(requests_per_minute=60_000)
    monkeypatch.setattr(downloader, "rate_limiter", limiter)
    monkeypatch.setattr(limiter, "acquire", lambda stop_event=None: True)
    monkeypatch.setattr(downloader.time, "sleep", delays.append)
    monkeypatch.setattr(downloader.random, "uniform", lambda start, end: 0.5)

    result = downloader.download_one(
        api,
        "train_landmark_files/123/retry.parquet",
        tmp_path,
    )

    assert result == ("downloaded", None)
    assert api.calls == 3
    assert delays == [2.5, 4.5]
    assert limiter.retry_count == 2


def test_429_threshold_prevents_queued_file_requests(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    class RateLimitError(Exception):
        def __init__(self) -> None:
            super().__init__("429")
            self.response = SimpleNamespace(status_code=429, headers={"Retry-After": "60"})

    class RateLimitedApi:
        def __init__(self) -> None:
            self.calls = 0

        def competition_download_file(
            self,
            competition: str,
            file_name: str,
            path: str,
            quiet: bool,
            unzip: bool,
            force: bool,
        ) -> None:
            self.calls += 1
            raise RateLimitError()

    api = RateLimitedApi()
    monkeypatch.setattr(
        downloader,
        "rate_limiter",
        downloader.GlobalRequestRateLimiter(requests_per_minute=60_000),
    )
    monkeypatch.setattr(downloader, "get_thread_kaggle_api", lambda: api)

    result = downloader.run_downloader(
        [
            "train_landmark_files/123/one.parquet",
            "train_landmark_files/123/two.parquet",
            "train_landmark_files/123/three.parquet",
        ],
        tmp_path,
        workers=1,
        progress_interval=1,
        max_429_events=1,
        min_free_bytes=0,
    )

    assert result == (0, 0, 1)
    assert api.calls == 1


def test_429_cooldown_stops_another_worker_waiting_for_a_request_slot(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    class RateLimitError(Exception):
        def __init__(self) -> None:
            super().__init__("429")
            self.response = SimpleNamespace(status_code=429, headers={"Retry-After": "60"})

    class RateLimitedApi:
        def __init__(self) -> None:
            self.calls = 0

        def competition_download_file(
            self,
            competition: str,
            file_name: str,
            path: str,
            quiet: bool,
            unzip: bool,
            force: bool,
        ) -> None:
            self.calls += 1
            raise RateLimitError()

    api = RateLimitedApi()
    limiter = downloader.GlobalRequestRateLimiter(requests_per_minute=60.0)
    monkeypatch.setattr(downloader, "rate_limiter", limiter)
    monkeypatch.setattr(downloader, "get_thread_kaggle_api", lambda: api)

    result = downloader.run_downloader(
        [
            "train_landmark_files/123/one.parquet",
            "train_landmark_files/123/two.parquet",
        ],
        tmp_path,
        workers=2,
        progress_interval=1,
        max_429_events=1,
        min_free_bytes=0,
    )

    assert result == (0, 0, 1)
    assert api.calls == 1
    assert limiter.event_count_429 == 1
