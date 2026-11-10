"""In-memory session manager with 30-minute idle TTL for stateful recognition tracking."""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from backend.app.core.config import get_settings
from backend.app.recognition.base import SignRecognizer
from backend.app.recognition.segmenter import SignSegmenter
from backend.app.recognition.stability import StabilityFilter
from backend.app.recognition.words import WordBuilder


class Session:
    """Encapsulates client state across streaming recognition frames."""

    def __init__(
        self, session_id: str, required_frames: int = 8, cooldown_frames: int = 10
    ) -> None:
        self.session_id = session_id
        self.stability_filter = StabilityFilter(
            required_frames=required_frames,
            cooldown_frames=cooldown_frames,
        )
        self.word_builder = WordBuilder()
        self.segmenter = SignSegmenter()
        self.processing_lock = threading.RLock()
        self.recognizers: dict[str, SignRecognizer] = {}
        self._recognizers_lock = threading.RLock()
        self.last_active: datetime = datetime.now(UTC)

    def get_recognizer(
        self, language: str, factory: Callable[[], SignRecognizer]
    ) -> SignRecognizer:
        """Return the cached recognizer for a normalized language, creating it once."""
        with self.processing_lock:
            with self._recognizers_lock:
                if language not in self.recognizers:
                    self.recognizers[language] = factory()
                return self.recognizers[language]

    def touch(self) -> None:
        """Update last active timestamp to current time."""
        self.last_active = datetime.now(UTC)

    def reset(self) -> None:
        """Reset stateful components within this session."""
        with self.processing_lock:
            with self._recognizers_lock:
                self.stability_filter.reset()
                self.word_builder.clear()
                self.segmenter.reset()
                self.recognizers.clear()
                self.touch()


class SessionManager:
    """Manages active user sessions in memory with idle TTL expiration."""

    def __init__(self, ttl_seconds: int | None = None) -> None:
        settings = get_settings()
        self.ttl = timedelta(
            seconds=ttl_seconds if ttl_seconds is not None else settings.session_ttl_seconds
        )
        self.stability_required_frames = settings.stability_required_frames
        self.stability_cooldown_frames = settings.stability_cooldown_frames
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> Session:
        """Retrieve existing active session or create a new one.

        Args:
            session_id: Unique client session identifier.

        Returns:
            Active Session instance.
        """
        with self._lock:
            self._cleanup_expired_locked()
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

            new_session = Session(
                session_id=session_id,
                required_frames=self.stability_required_frames,
                cooldown_frames=self.stability_cooldown_frames,
            )
            self._sessions[session_id] = new_session
            return new_session

    def reset(self, session_id: str) -> None:
        """Reset state for a given session.

        Args:
            session_id: Session identifier.
        """
        session = self._acquire_session(session_id, create=False)
        if session is None:
            return
        try:
            session.reset()
        finally:
            session.processing_lock.release()

    @contextmanager
    def locked_session(self, session_id: str) -> Iterator[Session]:
        """Yield the current session while holding its processing lock."""
        session = self._acquire_session(session_id, create=True)
        assert session is not None
        try:
            yield session
        finally:
            session.touch()
            session.processing_lock.release()

    def _acquire_session(self, session_id: str, *, create: bool) -> Session | None:
        """Acquire a still-current session, retrying after waits outside the map lock."""
        while True:
            with self._lock:
                self._cleanup_expired_locked()
                session = self._sessions.get(session_id)
                if session is None:
                    if not create:
                        return None
                    session = Session(
                        session_id=session_id,
                        required_frames=self.stability_required_frames,
                        cooldown_frames=self.stability_cooldown_frames,
                    )
                    self._sessions[session_id] = session

                if session.processing_lock.acquire(blocking=False):
                    return session

            session.processing_lock.acquire()
            session.processing_lock.release()

    def cleanup_expired(self) -> int:
        """Purge sessions inactive longer than the configured TTL.

        Returns:
            Count of evicted sessions.
        """
        with self._lock:
            return self._cleanup_expired_locked()

    def _cleanup_expired_locked(self) -> int:
        now = datetime.now(UTC)
        expired_count = 0
        for session_id, session in list(self._sessions.items()):
            if now - session.last_active > self.ttl and session.processing_lock.acquire(
                blocking=False
            ):
                try:
                    if (
                        self._sessions.get(session_id) is session
                        and now - session.last_active > self.ttl
                    ):
                        del self._sessions[session_id]
                        expired_count += 1
                finally:
                    session.processing_lock.release()
        return expired_count

    def count(self) -> int:
        """Return count of active sessions."""
        with self._lock:
            return len(self._sessions)


# Global default SessionManager
default_session_manager = SessionManager()
