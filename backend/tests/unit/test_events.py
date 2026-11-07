"""Unit tests for EventBus observer pattern."""

from backend.app.core.events import EventBus


def test_event_bus_subscribe_and_publish() -> None:
    """Verify published events reach subscribed handler."""
    bus = EventBus()
    received = []

    def handler(payload: dict) -> None:
        received.append(payload)

    bus.subscribe("word_committed", handler)
    bus.publish("word_committed", {"word": "HELLO", "session_id": "test"})

    assert len(received) == 1
    assert received[0] == {"word": "HELLO", "session_id": "test"}


def test_event_bus_unsubscribe() -> None:
    """Verify unsubscribed handlers are not called."""
    bus = EventBus()
    received = []

    def handler(payload: dict) -> None:
        received.append(payload)

    bus.subscribe("letter_confirmed", handler)
    bus.unsubscribe("letter_confirmed", handler)
    bus.publish("letter_confirmed", {"letter": "A"})

    assert len(received) == 0


def test_event_bus_multiple_listeners() -> None:
    """Verify multiple subscribers to same event both receive payloads."""
    bus = EventBus()
    res1, res2 = [], []

    bus.subscribe("test_event", lambda p: res1.append(p))
    bus.subscribe("test_event", lambda p: res2.append(p))

    bus.publish("test_event", {"val": 42})
    assert len(res1) == 1
    assert len(res2) == 1


def test_event_bus_clear() -> None:
    """Verify clear removes all subscribers."""
    bus = EventBus()
    res = []
    bus.subscribe("event", lambda p: res.append(p))
    bus.clear()
    bus.publish("event", {"data": 1})
    assert len(res) == 0
