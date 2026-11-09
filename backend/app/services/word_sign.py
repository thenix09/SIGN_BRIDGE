"""Word-level sequence recognition service."""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.core.events import EventBus, default_event_bus
from backend.app.domain.gesture import Gesture
from backend.app.domain.sequence import LandmarkFrame, LandmarkWindow
from backend.app.recognition.base import WordRecognizer
from backend.app.recognition.factory import RecognizerFactory
from backend.app.recognition.segmenter import SegmenterState
from backend.app.services.sessions import SessionManager, default_session_manager


@dataclass(frozen=True)
class WordResult:
    """Result of word recognition on streaming frame or completed window."""

    state: str  # "WAITING" / "SIGNING" / "RECOGNISING" / "COMPLETE"
    top_k: list[Gesture]
    accepted_word: str | None
    sentence: str
    message: str | None


class WordSignService:
    """Orchestrates sequence sign segmentation, recognition, and sentence accumulation."""

    def __init__(
        self,
        factory: type[RecognizerFactory] | None = None,
        sessions: SessionManager | None = None,
        bus: EventBus | None = None,
        confidence_threshold: float = 0.50,
        margin_threshold: float = 0.10,
    ) -> None:
        self.factory = factory or RecognizerFactory
        self.sessions = sessions or default_session_manager
        self.bus = bus or default_event_bus
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

        # Cached instances per language code
        self._recognizers: dict[str, WordRecognizer] = {}

    def _get_recognizer(self, language: str) -> WordRecognizer:
        """Retrieve cached WordRecognizer instance for the requested language."""
        lang_key = language.upper()
        if lang_key not in self._recognizers:
            self._recognizers[lang_key] = self.factory.create_word(lang_key)
        return self._recognizers[lang_key]

    def set_recognizer(self, language: str, recognizer: WordRecognizer) -> None:
        """Inject an explicit recognizer for testing purposes."""
        self._recognizers[language.upper()] = recognizer

    def recognize_window(
        self, session_id: str, language: str, window: LandmarkWindow
    ) -> WordResult:
        """Recognize an isolated sign gesture from a completed LandmarkWindow.

        Args:
            session_id: Client session identifier.
            language: Sign language code (e.g. 'ASL').
            window: Completed LandmarkWindow of frames.

        Returns:
            WordResult with top_k candidates and accepted word if confident.
        """
        session = self.sessions.get(session_id)
        recognizer = self._get_recognizer(language)
        top_k = recognizer.recognize(window, top_k=3)

        accepted_word: str | None = None
        message: str | None = None

        if top_k:
            top_1 = top_k[0]
            top_2_conf = top_k[1].confidence if len(top_k) > 1 else 0.0

            meets_threshold = top_1.confidence >= self.confidence_threshold
            meets_margin = (top_1.confidence - top_2_conf) >= self.margin_threshold

            if meets_threshold and meets_margin:
                accepted_word = top_1.label
                session.word_builder.commit_word(accepted_word)
                message = None

                self.bus.publish(
                    "word_recognised",
                    {
                        "session_id": session_id,
                        "word": accepted_word,
                        "language": language,
                        "confidence": top_1.confidence,
                    },
                )
            else:
                accepted_word = None
                message = "No confident match, please try again"
        else:
            message = "No confident match, please try again"

        return WordResult(
            state="COMPLETE",
            top_k=top_k,
            accepted_word=accepted_word,
            sentence=session.word_builder.sentence,
            message=message,
        )

    def process_frame(
        self,
        session_id: str,
        language: str,
        frame: LandmarkFrame | None,
    ) -> WordResult:
        """Process a streaming frame through session segmenter and evaluate when complete.

        Args:
            session_id: Client session identifier.
            language: Sign language code (e.g. 'ASL').
            frame: Single LandmarkFrame or None if no detections.

        Returns:
            WordResult reflecting current segmenter or recognition state.
        """
        session = self.sessions.get(session_id)
        seg_state, completed_window = session.segmenter.update(frame)

        if seg_state == SegmenterState.COMPLETE and completed_window is not None:
            return self.recognize_window(session_id, language, completed_window)

        if seg_state == SegmenterState.SIGNING:
            return WordResult(
                state="SIGNING",
                top_k=[],
                accepted_word=None,
                sentence=session.word_builder.sentence,
                message=None,
            )

        # Segmenter is IDLE / WAITING
        return WordResult(
            state="WAITING",
            top_k=[],
            accepted_word=None,
            sentence=session.word_builder.sentence,
            message=None,
        )
