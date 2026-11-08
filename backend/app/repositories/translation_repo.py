"""SQLite repository implementation for Translation domain entities."""

from datetime import UTC, datetime
from typing import Literal, cast

from sqlalchemy import desc, select
from sqlalchemy.orm import sessionmaker

from backend.app.domain.translation import Translation
from backend.app.repositories.base import Repository
from backend.app.repositories.db import TranslationModel, create_session_factory


class SQLiteTranslationRepository(Repository[Translation]):
    """SQLAlchemy/SQLite backed repository for Translation entities."""

    def __init__(self, session_factory: sessionmaker | None = None) -> None:
        self._session_factory = session_factory or create_session_factory()

    def add(self, item: Translation) -> Translation:
        """Persist a new Translation item and return domain entity with assigned id."""
        with self._session_factory() as session:
            model = TranslationModel(
                direction=item.direction,
                language=item.language,
                content=item.content,
                created_at=item.created_at or datetime.now(UTC),
            )
            session.add(model)
            session.commit()
            session.refresh(model)

            return Translation(
                id=model.id,
                direction=cast(Literal["sign_to_text", "text_to_sign"], model.direction),
                language=model.language,
                content=model.content,
                created_at=model.created_at,
            )

    def get(self, id_: int) -> Translation | None:
        """Retrieve a Translation by its primary key id."""
        with self._session_factory() as session:
            model = session.get(TranslationModel, id_)
            if model is None:
                return None
            return Translation(
                id=model.id,
                direction=cast(Literal["sign_to_text", "text_to_sign"], model.direction),
                language=model.language,
                content=model.content,
                created_at=model.created_at,
            )

    def list(self, limit: int = 100) -> list[Translation]:
        """List translations ordered by created_at descending up to limit."""
        with self._session_factory() as session:
            stmt = (
                select(TranslationModel)
                .order_by(desc(TranslationModel.created_at), desc(TranslationModel.id))
                .limit(limit)
            )
            models = session.scalars(stmt).all()
            return [
                Translation(
                    id=m.id,
                    direction=cast(Literal["sign_to_text", "text_to_sign"], m.direction),
                    language=m.language,
                    content=m.content,
                    created_at=m.created_at,
                )
                for m in models
            ]

    def clear(self) -> None:
        """Delete all translation records."""
        with self._session_factory() as session:
            session.query(TranslationModel).delete()
            session.commit()
