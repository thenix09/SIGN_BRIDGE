"""Unit tests for the text-to-sign rendering subsystem."""

import pytest

from backend.app.core.events import EventBus
from backend.app.rendering.base import SignFrame
from backend.app.rendering.factory import RendererFactory
from backend.app.rendering.renderers import (
    CHEST_ANCHOR_X,
    CHEST_ANCHOR_Y,
    FallbackSignRenderer,
    LandmarkTemplateRenderer,
    WordSequenceRenderer,
)
from backend.app.rendering.stores import LexiconStore, TemplateStore
from backend.app.services.text_to_sign import TextToSignService


def test_template_store_loading_and_retrieval() -> None:
    store = TemplateStore()
    assert store.has("A")
    assert store.has("Z")
    assert not store.has("1")
    assert not store.has("?")

    template_a = store.get("A")
    assert len(template_a) == 21
    assert len(template_a[0]) >= 2

    with pytest.raises(KeyError):
        store.get("INVALID")


def test_landmark_template_renderer_anchoring() -> None:
    renderer = LandmarkTemplateRenderer()
    frame = renderer.render_char("A")
    assert frame is not None
    assert frame.label == "A"
    assert frame.kind == "letter"
    assert frame.pose == []
    assert frame.left_hand is None
    assert frame.right_hand is not None
    assert len(frame.right_hand) == 21

    # Wrist coordinate in template is [0.0, 0.0, 0.0] -> anchored to CHEST_ANCHOR
    wrist_pt = frame.right_hand[0]
    assert abs(wrist_pt[0] - CHEST_ANCHOR_X) < 1e-4
    assert abs(wrist_pt[1] - CHEST_ANCHOR_Y) < 1e-4

    # Multiple characters
    frames = renderer.render("AB")
    assert len(frames) == 2
    assert [f.label for f in frames] == ["A", "B"]


def test_fallback_sign_renderer_tokenization_and_metadata() -> None:
    # Setup mock lexicon with word 'hello'
    lexicon_store = LexiconStore()
    lexicon_store._lexicon["hello"] = [
        SignFrame(
            label="hello",
            kind="word",
            pose=[[0.5, 0.5] for _ in range(9)],
            left_hand=None,
            right_hand=[[0.5, 0.5] for _ in range(21)],
            duration_ms=300,
        )
    ]
    word_renderer = WordSequenceRenderer(lexicon_store=lexicon_store)
    template_renderer = LandmarkTemplateRenderer()
    fallback = FallbackSignRenderer(
        word_renderer=word_renderer,
        template_renderer=template_renderer,
        aliases={"thanks": "thankyou", "thank you": "thankyou"},
    )

    # Text: "Hello CAT!" -> "hello" is word, "cat" falls back to spelled (C, A, T)
    frames, segments, fallbacks = fallback.render_with_metadata("Hello, CAT!")

    assert fallbacks == ["cat"]
    assert len(segments) == 2
    assert segments[0]["token"] == "hello"
    assert segments[0]["kind"] == "word"
    assert segments[1]["token"] == "cat"
    assert segments[1]["kind"] == "spelled"

    # Frame sequence: 1 word frame + 1 pause frame + 3 letter frames (C, A, T)
    assert len(frames) == 5
    assert frames[0].label == "hello"
    assert frames[0].kind == "word"
    assert frames[1].kind == "pause"
    assert [f.label for f in frames[2:]] == ["C", "A", "T"]


def test_fallback_sign_renderer_alias_mapping() -> None:
    lexicon_store = LexiconStore()
    lexicon_store._lexicon["thankyou"] = [
        SignFrame(
            label="thankyou",
            kind="word",
            pose=[],
            left_hand=None,
            right_hand=None,
            duration_ms=300,
        )
    ]
    word_renderer = WordSequenceRenderer(lexicon_store=lexicon_store)
    fallback = FallbackSignRenderer(
        word_renderer=word_renderer,
        aliases={"thank you": "thankyou"},
    )

    frames, segments, fallbacks = fallback.render_with_metadata("Thank you!")
    assert fallbacks == []
    assert len(segments) == 1
    assert segments[0]["token"] == "thankyou"
    assert segments[0]["kind"] == "word"


def test_renderer_factory_modes() -> None:
    auto_r = RendererFactory.create("auto")
    assert isinstance(auto_r, FallbackSignRenderer)

    letters_r = RendererFactory.create("letters")
    assert isinstance(letters_r, LandmarkTemplateRenderer)

    words_r = RendererFactory.create("words")
    assert isinstance(words_r, WordSequenceRenderer)


def test_text_to_sign_service_and_event_publishing() -> None:
    bus = EventBus()
    events: list[dict] = []
    bus.subscribe("text_converted", lambda payload: events.append(payload))

    service = TextToSignService(bus=bus)
    result = service.convert("ASL", "HI")

    assert len(result.frames) == 2
    assert [f.label for f in result.frames] == ["H", "I"]
    assert len(events) == 1
    assert events[0]["text"] == "HI"
    assert events[0]["language"] == "ASL"


def test_text_to_sign_service_length_validation() -> None:
    service = TextToSignService()
    too_long = "A" * 201
    with pytest.raises(ValueError, match="exceeds maximum allowed limit of 200"):
        service.convert("ASL", too_long)
