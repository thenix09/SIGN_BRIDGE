"""Database configuration, SQLAlchemy 2.x declarative models and session management."""

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Engine,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)

from backend.app.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base class for SQLAlchemy ORM models."""

    pass


class TranslationModel(Base):
    """SQLAlchemy model for translations."""

    __tablename__ = "translations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    direction: Mapped[str] = mapped_column(String(32), nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class SampleModel(Base):
    """SQLAlchemy model for letter samples collected for training."""

    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    group: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    features: Mapped[str] = mapped_column(Text, nullable=False)  # JSON or comma-separated floats
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create and return a SQLAlchemy database engine."""
    url = database_url or get_settings().database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def create_session_factory(engine: Engine | None = None) -> sessionmaker:
    """Create a sessionmaker bound to the given engine."""
    eng = engine or create_db_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=eng)


def init_db(engine: Engine | None = None) -> None:
    """Initialize database tables according to declarative metadata."""
    eng = engine or create_db_engine()
    Base.metadata.create_all(bind=eng)
