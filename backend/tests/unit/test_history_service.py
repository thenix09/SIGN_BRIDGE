"""Unit tests for HistoryService and EventBus observer integration."""

from datetime import UTC, datetime

import pytest

from backend.app.core.events import EventBus
from backend.app.domain.translation import Translation
from backend.app.repositories.db import (
    create_db_engine,
    create_session_factory,
    init_db,
)
from backend.app.repositories.translation_repo import SQLiteTranslationRepository
from backend.app.services.history import HistoryService


@pytest.fixture
def history_service():
    """Create a HistoryService backed by in-memory SQLite and a fresh EventBus."""
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = create_session_factory(engine)
    repo = SQLiteTranslationRepository(session_factory=session_factory)
    bus = EventBus()
    service = HistoryService(repo=repo, bus=bus)
    return service, bus


def test_history_service_direct_record(history_service) -> None:
    """Verify manual record and list_recent."""
    service, _ = history_service
    t = Translation(
        id=None,
        direction="sign_to_text",
        language="ASL",
        content="TEST",
        created_at=datetime.now(UTC),
    )
    saved = service.record(t)
    assert saved.id is not None
    assert saved.content == "TEST"

    recent = service.list_recent(limit=10)
    assert len(recent) == 1
    assert recent[0].content == "TEST"

    service.clear()
    assert service.list_recent() == []


def test_event_bus_word_committed_creates_history(history_service) -> None:
    """Verify publishing 'word_committed' automatically persists a history translation."""
    service, bus = history_service

    bus.publish("word_committed", {"word": "HELLO", "language": "ASL"})

    recent = service.list_recent()
    assert len(recent) == 1
    assert recent[0].content == "HELLO"
    assert recent[0].direction == "sign_to_text"
    assert recent[0].language == "ASL"


def test_event_bus_word_recognised_creates_history(history_service) -> None:
    """Verify publishing 'word_recognised' automatically persists a history translation."""
    service, bus = history_service

    bus.publish("word_recognised", {"word": "WORLD", "language": "ASL"})

    recent = service.list_recent()
    assert len(recent) == 1
    assert recent[0].content == "WORLD"
    assert recent[0].direction == "sign_to_text"


def test_event_bus_text_converted_creates_history(history_service) -> None:
    """Verify publishing 'text_converted' automatically persists a history translation."""
    service, bus = history_service

    bus.publish("text_converted", {"text": "PLEASE", "language": "ASL"})

    recent = service.list_recent()
    assert len(recent) == 1
    assert recent[0].content == "PLEASE"
    assert recent[0].direction == "text_to_sign"


def test_event_bus_empty_payload_ignored(history_service) -> None:
    """Verify empty or missing word/text in payload is safely ignored."""
    service, bus = history_service

    bus.publish("word_committed", {})
    bus.publish("word_recognised", {"word": ""})
    bus.publish("text_converted", {"text": ""})

    assert service.list_recent() == []
