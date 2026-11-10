"""Tests for session-owned sign recognizers and shared model resources."""

import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from backend.app.core.events import EventBus
from backend.app.core.exceptions import InvalidLandmarkError, UnsupportedLanguageError
from backend.app.recognition import landmark_recognizer
from backend.app.recognition.asl_recognizer import ASLRecognizer
from backend.app.recognition.base import FakeRecognizer, SignRecognizer
from backend.app.recognition.factory import RecognizerFactory
from backend.app.recognition.words import SuggestionEngine
from backend.app.services.sessions import SessionManager
from backend.app.services.sign_to_text import RecognitionResult, SignToTextService


class GatedRLock:
    """Reentrant lock with an event gate for deterministic acquisition races."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.waiting = threading.Event()
        self.allow_waiter = threading.Event()
        self._gate_next_wait = False

    def gate_next_wait(self) -> None:
        self._gate_next_wait = True

    def acquire(self, blocking: bool = True) -> bool:
        if blocking and self._gate_next_wait:
            self._gate_next_wait = False
            self.waiting.set()
            if not self.allow_waiter.wait(timeout=5):
                raise TimeoutError("Timed out waiting to release gated lock acquisition")
        return self._lock.acquire(blocking=blocking)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "GatedRLock":
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


class CountingFactory(RecognizerFactory):
    """Factory tracking session-scoped recognizer creation for tests."""

    calls: list[str] = []

    @classmethod
    def reset(cls) -> None:
        cls.calls = []

    @classmethod
    def create(cls, language: str) -> SignRecognizer:
        cls.calls.append(language)
        language_key = language.upper()
        if language_key not in {"ASL", "ISL"}:
            return super().create(language)
        return FakeRecognizer(
            language=language_key,
            default_label=language_key[0],
            default_confidence=0.95,
        )


class ProbabilityModel:
    """Deterministic model whose probabilities follow a frame feature."""

    classes_ = ["A", "B"]

    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.load_calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.lock = threading.Lock()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        with self.lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.delay:
                time.sleep(self.delay)
            if features[0, 4] > 0:
                return np.array([[0.9, 0.1]])
            return np.array([[0.1, 0.9]])
        finally:
            with self.lock:
                self.active_calls -= 1


def _build_service(
    factory: type[RecognizerFactory],
    sessions: SessionManager | None = None,
) -> SignToTextService:
    return SignToTextService(
        factory=factory,
        sessions=sessions or SessionManager(ttl_seconds=300),
        suggestions=SuggestionEngine(["A", "I"]),
    )


def _landmarks(signal: float = 1.0) -> list[list[float]]:
    points = [[0.5, 0.5, 0.0] for _ in range(21)]
    points[1] = [0.5, 0.5 + signal * 0.2, 0.0]
    return points


def _process_until_confirmed(
    service: SignToTextService, session_id: str, language: str = "ASL"
) -> RecognitionResult:
    service.process_frame(session_id, language, _landmarks())
    return service.process_frame(session_id, language, _landmarks())


@pytest.fixture
def service() -> SignToTextService:
    CountingFactory.reset()
    return _build_service(CountingFactory)


def test_recognizer_cache_is_per_session_and_normalized_language(
    service: SignToTextService,
) -> None:
    service.process_frame("session-one", "asl", _landmarks())
    first_session_recognizer = service.sessions.get("session-one").recognizers["ASL"]
    service.process_frame("session-one", "ASL", _landmarks())
    service.process_frame("session-two", "ASL", _landmarks())

    first = service.sessions.get("session-one").recognizers
    second = service.sessions.get("session-two").recognizers
    assert set(first) == {"ASL"}
    assert set(second) == {"ASL"}
    assert first["ASL"] is first_session_recognizer
    assert first["ASL"] is not second["ASL"]
    assert CountingFactory.calls == ["asl", "ASL"]


def test_different_languages_have_separate_recognizers_within_session(
    service: SignToTextService,
) -> None:
    service.process_frame("session", "ASL", _landmarks())
    service.process_frame("session", "ISL", _landmarks())

    recognizers = service.sessions.get("session").recognizers
    assert set(recognizers) == {"ASL", "ISL"}
    assert recognizers["ASL"] is not recognizers["ISL"]
    assert CountingFactory.calls == ["ASL", "ISL"]


def test_session_stability_and_word_state_remain_independent(
    service: SignToTextService,
) -> None:
    service.sessions.stability_required_frames = 2
    service.process_frame("session-one", "ASL", _landmarks())
    confirmed = service.process_frame("session-one", "ASL", _landmarks())
    other_session = service.process_frame("session-two", "ASL", _landmarks())

    assert confirmed.confirmed_letter == "A"
    assert confirmed.current_word == "A"
    assert other_session.confirmed_letter is None
    assert other_session.state == "DETECTING"
    assert other_session.current_word == ""


def test_empty_invalid_and_unsupported_frames_preserve_existing_behavior(
    service: SignToTextService,
) -> None:
    empty = service.process_frame("session-empty", "UNSUPPORTED", None)
    assert empty.label == ""
    assert CountingFactory.calls == []

    with pytest.raises(InvalidLandmarkError):
        service.process_frame("session-invalid", "ASL", _landmarks()[:10])
    assert CountingFactory.calls == []

    with pytest.raises(
        UnsupportedLanguageError,
        match="Sign language 'unsupported' is not supported",
    ):
        service.process_frame("session-unsupported", "unsupported", _landmarks())
    assert CountingFactory.calls == ["unsupported"]


def test_session_reset_and_expiry_release_recognizer_state(
    service: SignToTextService,
) -> None:
    service.process_frame("session-reset", "ASL", _landmarks())
    reset_session = service.sessions.get("session-reset")
    old_recognizer = reset_session.recognizers["ASL"]

    service.reset_session("session-reset")
    assert reset_session.recognizers == {}
    service.process_frame("session-reset", "ASL", _landmarks())
    assert reset_session.recognizers["ASL"] is not old_recognizer

    service.process_frame("session-expired", "ASL", _landmarks())
    expired_session = service.sessions.get("session-expired")
    recognizer_ref = weakref.ref(expired_session.recognizers["ASL"])
    expired_session.last_active = datetime.now(UTC) - timedelta(hours=1)
    assert service.sessions.cleanup_expired() == 1
    del expired_session
    gc.collect()
    assert recognizer_ref() is None


def test_cleanup_skips_busy_session_then_expires_it_when_idle(
    service: SignToTextService,
) -> None:
    session = service.sessions.get("busy-session")
    session.last_active = datetime.now(UTC) - timedelta(hours=1)
    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_session_lock() -> None:
        with session.processing_lock:
            lock_acquired.set()
            assert release_lock.wait(timeout=5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        hold_future = executor.submit(hold_session_lock)
        assert lock_acquired.wait(timeout=5)
        assert service.sessions.cleanup_expired() == 0
        assert service.sessions.count() == 1
        release_lock.set()
        hold_future.result(timeout=5)

    assert service.sessions.cleanup_expired() == 1
    assert service.sessions.count() == 0


def test_waiting_frame_retries_if_cleanup_expires_its_original_session(
    service: SignToTextService,
) -> None:
    manager = service.sessions
    old_session = manager.get("expiring-session")
    old_session.last_active = datetime.now(UTC) - timedelta(hours=1)
    gated_lock = GatedRLock()
    old_session.processing_lock = gated_lock  # type: ignore[assignment]
    gated_lock.acquire()
    gated_lock.gate_next_wait()

    with ThreadPoolExecutor(max_workers=1) as executor:
        frame_future = executor.submit(
            service.process_frame,
            "expiring-session",
            "ASL",
            _landmarks(),
        )
        assert gated_lock.waiting.wait(timeout=5)

        gated_lock.release()
        assert manager.cleanup_expired() == 1
        assert manager.count() == 0

        gated_lock.allow_waiter.set()
        frame_result = frame_future.result(timeout=5)

    frame_session = manager.get("expiring-session")
    assert frame_result.label == "A"
    assert frame_session is not old_session
    assert frame_session.recognizers["ASL"] is not old_session.recognizers.get("ASL")


def test_interleaved_sessions_keep_ema_smoothing_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "asl_model.joblib"
    model_path.touch()
    model = ProbabilityModel()
    monkeypatch.setattr(landmark_recognizer.joblib, "load", lambda _path: model)

    class ModelFactory(RecognizerFactory):
        @classmethod
        def create(cls, language: str) -> SignRecognizer:
            if language.upper() == "ASL":
                return ASLRecognizer(model_path=model_path, threshold=0.0)
            return super().create(language)

    service = _build_service(ModelFactory)
    s1_first = service.process_frame("s1", "ASL", _landmarks(1.0))
    s2_first = service.process_frame("s2", "ASL", _landmarks(-1.0))
    s1_second = service.process_frame("s1", "ASL", _landmarks(1.0))

    assert s1_first.label == "A"
    assert s2_first.label == "B"
    assert s1_second.label == "A"
    assert s1_first.confidence == s2_first.confidence == s1_second.confidence == 0.9


def test_model_load_and_inference_are_shared_and_thread_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "asl_model.joblib"
    model_path.touch()
    model = ProbabilityModel(delay=0.02)
    load_calls = 0
    load_lock = threading.Lock()

    def load_model(_path: Path) -> ProbabilityModel:
        nonlocal load_calls
        with load_lock:
            load_calls += 1
        return model

    monkeypatch.setattr(landmark_recognizer.joblib, "load", load_model)

    class ModelFactory(RecognizerFactory):
        @classmethod
        def create(cls, language: str) -> SignRecognizer:
            if language.upper() == "ASL":
                return ASLRecognizer(model_path=model_path, threshold=0.0)
            return super().create(language)

    service = _build_service(ModelFactory)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                service.process_frame,
                f"session-{index}",
                "ASL",
                _landmarks(1.0 if index % 2 == 0 else -1.0),
            )
            for index in range(16)
        ]
        results = [future.result() for future in futures]

    recognizers = [
        cast(
            landmark_recognizer.LandmarkRecognizer,
            service.sessions.get(f"session-{index}").recognizers["ASL"],
        )
        for index in range(16)
    ]
    assert all(result.label in {"A", "B"} for result in results)
    assert len({id(recognizer) for recognizer in recognizers}) == 16
    assert all(recognizer._resource is recognizers[0]._resource for recognizer in recognizers)
    assert load_calls == 1
    assert model.max_active_calls == 1


def test_reset_waits_for_in_flight_frame_and_next_frame_sees_clean_state() -> None:
    entered_publication = threading.Event()
    release_publication = threading.Event()
    reset_started = threading.Event()
    bus = EventBus()
    service = _build_service(CountingFactory)
    service.bus = bus
    session = service.sessions.get("reset-race")
    session.stability_filter.required_frames = 2

    def block_first_publication(payload: dict[str, object]) -> None:
        if payload["session_id"] == "reset-race" and not entered_publication.is_set():
            entered_publication.set()
            assert release_publication.wait(timeout=5)

    bus.subscribe("letter_confirmed", block_first_publication)

    with ThreadPoolExecutor(max_workers=2) as executor:
        frame_future = executor.submit(
            _process_until_confirmed, service, "reset-race"
        )
        assert entered_publication.wait(timeout=5)

        def reset() -> None:
            reset_started.set()
            service.reset_session("reset-race")

        reset_future = executor.submit(reset)
        assert reset_started.wait(timeout=5)
        assert not reset_future.done()

        release_publication.set()
        in_flight_result = frame_future.result(timeout=5)
        reset_future.result(timeout=5)

    assert in_flight_result.confirmed_letter == "A"
    assert in_flight_result.current_word == "A"
    assert session.word_builder.current_word == ""
    assert session.stability_filter.state.value == "IDLE"
    assert session.recognizers == {}

    bus.unsubscribe("letter_confirmed", block_first_publication)
    next_result = _process_until_confirmed(service, "reset-race")
    assert next_result.confirmed_letter == "A"
    assert next_result.current_word == "A"


def test_commit_and_backspace_wait_for_frame_and_keep_state_consistent() -> None:
    entered_publication = threading.Event()
    release_publication = threading.Event()
    commit_started = threading.Event()
    backspace_started = threading.Event()
    bus = EventBus()
    service = _build_service(CountingFactory)
    service.bus = bus
    session = service.sessions.get("word-race")
    session.stability_filter.required_frames = 2

    def block_publication(payload: dict[str, object]) -> None:
        if payload["session_id"] == "word-race":
            entered_publication.set()
            assert release_publication.wait(timeout=5)

    bus.subscribe("letter_confirmed", block_publication)

    with ThreadPoolExecutor(max_workers=3) as executor:
        frame_future = executor.submit(
            _process_until_confirmed, service, "word-race"
        )
        assert entered_publication.wait(timeout=5)

        def commit() -> None:
            commit_started.set()
            service.commit_word("word-race")

        def backspace() -> None:
            backspace_started.set()
            service.backspace("word-race")

        commit_future = executor.submit(commit)
        backspace_future = executor.submit(backspace)
        assert commit_started.wait(timeout=5)
        assert backspace_started.wait(timeout=5)
        assert not commit_future.done()
        assert not backspace_future.done()

        release_publication.set()
        frame_result = frame_future.result(timeout=5)
        commit_future.result(timeout=5)
        backspace_future.result(timeout=5)

    assert frame_result.confirmed_letter == "A"
    assert frame_result.current_word == "A"
    assert frame_result.sentence == "A"
    assert (session.word_builder.current_word, session.word_builder.sentence) in {
        ("A", "A"),
        ("", ""),
    }


def test_distinct_sessions_process_in_parallel() -> None:
    bus = EventBus()
    service = _build_service(CountingFactory)
    service.bus = bus
    entered_sessions = {"parallel-one": threading.Event(), "parallel-two": threading.Event()}
    release_frames = threading.Event()

    def block_publication(payload: dict[str, object]) -> None:
        entered_sessions[str(payload["session_id"])].set()
        assert release_frames.wait(timeout=5)

    bus.subscribe("letter_confirmed", block_publication)
    for session_id in entered_sessions:
        service.sessions.get(session_id).stability_filter.required_frames = 2

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_process_until_confirmed, service, session_id)
            for session_id in entered_sessions
        ]
        assert all(event.wait(timeout=5) for event in entered_sessions.values())
        release_frames.set()
        results = [future.result(timeout=5) for future in futures]

    assert all(result.confirmed_letter == "A" for result in results)
