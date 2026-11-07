"""Unit tests for SQLite repositories using in-memory database."""

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.app.domain.sample import Sample
from backend.app.domain.translation import Translation
from backend.app.repositories.db import (
    create_db_engine,
    create_session_factory,
    init_db,
)
from backend.app.repositories.sample_repo import SQLiteSampleRepository
from backend.app.repositories.translation_repo import SQLiteTranslationRepository


@pytest.fixture
def session_factory():
    """Create in-memory SQLite database and return session factory."""
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    return create_session_factory(engine)


def test_translation_repository_crud(session_factory) -> None:
    """Verify add, get, list, and clear on SQLiteTranslationRepository."""
    repo = SQLiteTranslationRepository(session_factory=session_factory)

    # Empty initially
    assert repo.list() == []

    # Add items
    t1 = repo.add(
        Translation(
            id=None,
            direction="sign_to_text",
            language="ASL",
            content="HELLO",
            created_at=datetime.now(UTC),
        )
    )
    assert t1.id is not None
    assert t1.content == "HELLO"

    t2 = repo.add(
        Translation(
            id=None,
            direction="text_to_sign",
            language="ASL",
            content="THANK YOU",
            created_at=datetime.now(UTC),
        )
    )
    assert t2.id is not None

    # Get by ID
    fetched = repo.get(t1.id)
    assert fetched is not None
    assert fetched.content == "HELLO"
    assert fetched.direction == "sign_to_text"

    # Non-existent ID
    assert repo.get(9999) is None

    # List with limit
    items = repo.list(limit=10)
    assert len(items) == 2
    # Most recent first
    assert items[0].content == "THANK YOU"
    assert items[1].content == "HELLO"

    # Test limit=1
    single = repo.list(limit=1)
    assert len(single) == 1
    assert single[0].content == "THANK YOU"

    # Clear
    repo.clear()
    assert repo.list() == []


def test_sample_repository_crud_and_export(session_factory, tmp_path: Path) -> None:
    """Verify sample persistence, stats, count_for_label and CSV export."""
    repo = SQLiteSampleRepository(session_factory=session_factory)

    feat_a = tuple([0.1 * (i + 1) for i in range(63)])
    feat_b = tuple([0.05 * (i + 1) for i in range(63)])

    repo.add(Sample(label="A", group="user1", features=feat_a))
    repo.add(Sample(label="A", group="user1", features=feat_a))
    repo.add(Sample(label="B", group="user2", features=feat_b))

    # Check counts
    assert repo.count_for_label("A") == 2
    assert repo.count_for_label("B") == 1
    assert repo.count_for_label("C") == 0

    counts, total = repo.get_stats()
    assert counts == {"A": 2, "B": 1}
    assert total == 3

    # Check list
    samples = repo.list(limit=10)
    assert len(samples) == 3

    # Export to CSV
    export_file = tmp_path / "exported_samples.csv"
    count = repo.export_csv(export_file)
    assert count == 3
    assert export_file.is_file()

    # Read back CSV to verify structure
    with open(export_file, encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 4  # header + 3 rows
        header = reader[0]
        assert header[0] == "label"
        assert header[1] == "group"
        assert header[2] == "f0"
        assert header[64] == "f62"

        row1 = reader[1]
        assert row1[0] == "A"
        assert row1[1] == "user1"
        assert len(row1) == 65

    # Clear
    repo.clear()
    assert repo.list() == []
    counts_empty, total_empty = repo.get_stats()
    assert counts_empty == {}
    assert total_empty == 0
