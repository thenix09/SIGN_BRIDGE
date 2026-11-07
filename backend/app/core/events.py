"""EventBus observer implementation for decoupled inter-service notifications."""

from collections import defaultdict
from collections.abc import Callable
from typing import Any


class EventBus:
    """Observer pattern event bus for publishing domain events to registered subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[dict[str, Any]], None]]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """Register a handler callback for a specific event topic.

        Args:
            event: The name of the event (e.g. 'word_committed').
            handler: Callable taking event payload dictionary.
        """
        if handler not in self._subscribers[event]:
            self._subscribers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable[[dict[str, Any]], None]) -> None:
        """Unregister a handler callback from a specific event topic.

        Args:
            event: The name of the event.
            handler: Callable to remove.
        """
        if handler in self._subscribers[event]:
            self._subscribers[event].remove(handler)

    def publish(self, event: str, payload: dict[str, Any]) -> None:
        """Publish an event with its payload to all registered handlers.

        Args:
            event: The event topic name.
            payload: Data dictionary representing the event.
        """
        for handler in list(self._subscribers.get(event, [])):
            handler(payload)

    def clear(self) -> None:
        """Clear all registered event subscribers."""
        self._subscribers.clear()


# Global default EventBus instance
default_event_bus = EventBus()
