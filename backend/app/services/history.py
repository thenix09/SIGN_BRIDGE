"""History service for auditing and recording translation interactions."""

from datetime import UTC, datetime
from typing import Any

from backend.app.core.events import EventBus
from backend.app.domain.translation import Translation
from backend.app.repositories.base import Repository


class HistoryService:
    """Manages translation history persistence and subscribes to EventBus translation events."""

    def __init__(self, repo: Repository[Translation], bus: EventBus | None = None) -> None:
        """Initialize HistoryService and optionally bind to EventBus topics.

        Args:
            repo: Repository for persisting Translation entities.
            bus: EventBus instance to subscribe to. If provided, automatically
                 registers handlers for 'word_committed', 'word_recognised',
                 and 'text_converted'.
        """
        self.repo = repo
        self._bus = bus
        if bus is not None:
            bus.subscribe("word_committed", self._on_word_committed)
            bus.subscribe("word_recognised", self._on_word_recognised)
            bus.subscribe("text_converted", self._on_text_converted)

    def record(self, t: Translation) -> Translation:
        """Record and persist a translation event.

        Args:
            t: Translation entity to persist.

        Returns:
            The saved Translation with assigned id.
        """
        return self.repo.add(t)

    def list_recent(self, limit: int = 50) -> list[Translation]:
        """List the most recent translation records.

        Args:
            limit: Maximum records to return (default 50).

        Returns:
            List of Translation objects ordered most recent first.
        """
        return self.repo.list(limit=limit)

    def clear(self) -> None:
        """Clear all translation history."""
        self.repo.clear()

    def _on_word_committed(self, payload: dict[str, Any]) -> None:
        """Event handler for 'word_committed' event (fingerspelling word commit)."""
        content = payload.get("word") or payload.get("content") or ""
        if not content:
            return
        language = payload.get("language", "ASL")
        translation = Translation(
            id=None,
            direction="sign_to_text",
            language=language,
            content=str(content),
            created_at=datetime.now(UTC),
        )
        self.record(translation)

    def _on_word_recognised(self, payload: dict[str, Any]) -> None:
        """Event handler for 'word_recognised' event (Words mode recognized sign)."""
        content = payload.get("word") or payload.get("content") or ""
        if not content:
            return
        language = payload.get("language", "ASL")
        translation = Translation(
            id=None,
            direction="sign_to_text",
            language=language,
            content=str(content),
            created_at=datetime.now(UTC),
        )
        self.record(translation)

    def _on_text_converted(self, payload: dict[str, Any]) -> None:
        """Event handler for 'text_converted' event (Text to Sign conversion)."""
        content = payload.get("text") or payload.get("content") or ""
        if not content:
            return
        language = payload.get("language", "ASL")
        translation = Translation(
            id=None,
            direction="text_to_sign",
            language=language,
            content=str(content),
            created_at=datetime.now(UTC),
        )
        self.record(translation)
