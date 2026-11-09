"""In-memory session manager with 30-minute idle TTL for stateful recognition tracking."""

from datetime import UTC, datetime, timedelta

from backend.app.core.config import get_settings
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
        self.last_active: datetime = datetime.now(UTC)

    def touch(self) -> None:
        """Update last active timestamp to current time."""
        self.last_active = datetime.now(UTC)

    def reset(self) -> None:
        """Reset stateful components within this session."""
        self.stability_filter.reset()
        self.word_builder.clear()
        self.segmenter.reset()
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

    def get(self, session_id: str) -> Session:
        """Retrieve existing active session or create a new one.

        Args:
            session_id: Unique client session identifier.

        Returns:
            Active Session instance.
        """
        self.cleanup_expired()

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
        if session_id in self._sessions:
            self._sessions[session_id].reset()

    def cleanup_expired(self) -> int:
        """Purge sessions inactive longer than the configured TTL.

        Returns:
            Count of evicted sessions.
        """
        now = datetime.now(UTC)
        expired_keys = [sid for sid, s in self._sessions.items() if now - s.last_active > self.ttl]
        for sid in expired_keys:
            del self._sessions[sid]
        return len(expired_keys)

    def count(self) -> int:
        """Return count of active sessions."""
        return len(self._sessions)


# Global default SessionManager
default_session_manager = SessionManager()
