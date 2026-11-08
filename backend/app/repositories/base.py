"""Generic repository abstraction for persistence operations."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Repository(ABC, Generic[T]):
    """Abstract base repository defining CRUD persistence operations."""

    @abstractmethod
    def add(self, item: T) -> T:
        """Persist a new entity item.

        Args:
            item: Domain entity instance to add.

        Returns:
            The saved entity with any generated fields populated.
        """
        pass

    @abstractmethod
    def get(self, id_: int) -> T | None:
        """Retrieve an entity by its identifier.

        Args:
            id_: Identifier of the entity.

        Returns:
            Domain entity instance if found, else None.
        """
        pass

    @abstractmethod
    def list(self, limit: int = 100) -> list[T]:
        """List recent entities up to a given limit.

        Args:
            limit: Maximum number of entities to return.

        Returns:
            List of domain entity instances.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all entity records from storage."""
        pass
