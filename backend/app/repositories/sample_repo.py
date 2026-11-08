"""SQLite repository implementation for collected Sample domain entities."""

import csv
import json
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from backend.app.domain.sample import Sample
from backend.app.repositories.base import Repository
from backend.app.repositories.db import SampleModel, create_session_factory


class SQLiteSampleRepository(Repository[Sample]):
    """SQLAlchemy/SQLite backed repository for letter training Sample entities."""

    def __init__(self, session_factory: sessionmaker | None = None) -> None:
        self._session_factory = session_factory or create_session_factory()

    def add(self, item: Sample) -> Sample:
        """Persist a new letter Sample to the database."""
        features_json = json.dumps(list(item.features))
        with self._session_factory() as session:
            model = SampleModel(
                label=item.label.upper(),
                group=item.group,
                features=features_json,
            )
            session.add(model)
            session.commit()
            return item

    def get(self, id_: int) -> Sample | None:
        """Retrieve a Sample by its integer primary key."""
        with self._session_factory() as session:
            model = session.get(SampleModel, id_)
            if model is None:
                return None
            features_list = json.loads(model.features)
            return Sample(
                label=model.label,
                group=model.group,
                features=tuple(features_list),
            )

    def list(self, limit: int = 100) -> list[Sample]:
        """List recent samples up to limit."""
        with self._session_factory() as session:
            stmt = select(SampleModel).order_by(SampleModel.id.desc()).limit(limit)
            models = session.scalars(stmt).all()
            return [
                Sample(
                    label=m.label,
                    group=m.group,
                    features=tuple(json.loads(m.features)),
                )
                for m in models
            ]

    def clear(self) -> None:
        """Delete all sample records."""
        with self._session_factory() as session:
            session.query(SampleModel).delete()
            session.commit()

    def count_for_label(self, label: str) -> int:
        """Count total samples collected for a specific letter label."""
        with self._session_factory() as session:
            stmt = select(func.count(SampleModel.id)).where(SampleModel.label == label.upper())
            count = session.scalar(stmt)
            return count or 0

    def get_stats(self) -> tuple[dict[str, int], int]:
        """Return counts per letter label and total sample count.

        Returns:
            Tuple of (counts_dict, total_count).
        """
        with self._session_factory() as session:
            stmt = select(SampleModel.label, func.count(SampleModel.id)).group_by(SampleModel.label)
            rows = session.execute(stmt).all()
            counts = {label: count for label, count in rows}
            total = sum(counts.values())
            return counts, total

    def export_csv(self, path: Path) -> int:
        """Export all recorded samples to a CSV file in standard dataset format.

        Format: label,group,f0,f1,...,f62

        Args:
            path: Target file path.

        Returns:
            Number of rows exported.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        header = ["label", "group"] + [f"f{i}" for i in range(63)]

        with self._session_factory() as session:
            stmt = select(SampleModel).order_by(SampleModel.id.asc())
            models = session.scalars(stmt).all()

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                count = 0
                for m in models:
                    feat = json.loads(m.features)
                    writer.writerow([m.label, m.group] + feat)
                    count += 1

            return count
