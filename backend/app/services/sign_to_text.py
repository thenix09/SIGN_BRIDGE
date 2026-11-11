"""SignToText business service coordinating single-frame fingerspelling recognition."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.core.events import EventBus
from backend.app.core.exceptions import InvalidLandmarkError
from backend.app.domain.hand_frame import HandFrame, canonicalise_landmarks
from backend.app.recognition.base import SignRecognizer
from backend.app.recognition.factory import RecognizerFactory
from backend.app.recognition.words import SuggestionEngine
from backend.app.services.sessions import Session, SessionManager


@dataclass(frozen=True)
class RecognitionResult:
    """Consolidated recognition and spelling state for a single frame."""

    label: str
    confidence: float
    state: str
    confirmed_letter: str | None
    current_word: str
    sentence: str
    suggestions: list[str]


class SignToTextService:
    """Processes incoming landmark frames for fingerspelling, applies stability filtering and word building."""

    def __init__(
        self,
        factory: type[RecognizerFactory] | None = None,
        sessions: SessionManager | None = None,
        suggestions: SuggestionEngine | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.factory = factory or RecognizerFactory
        self.sessions = sessions or SessionManager()
        self.suggestions = suggestions or SuggestionEngine("backend/data/words.txt")
        self.bus = bus
        self._canonical_handedness = self._read_canonical_handedness()

    def _get_recognizer(self, session: Session, language: str) -> SignRecognizer:
        """Return the recognizer cached for this session and normalized language."""
        language_key = language.upper()
        return session.get_recognizer(
            language_key,
            lambda: self.factory.create(language),
        )

    def _read_canonical_handedness(self) -> str:
        """Read majority dataset handedness from letters_meta.json, defaulting to 'Right'."""
        meta_path = Path(get_settings().letters_meta_path)
        if meta_path.is_file():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("canonical_handedness", "Right")
            except Exception:
                pass
        return "Right"

    def process_frame(
        self,
        session_id: str,
        language: str,
        raw_landmarks: Sequence[Sequence[float]] | None,
        handedness: str = "Right",
    ) -> RecognitionResult:
        """Process a single video frame of hand landmarks.

        Args:
            session_id: Client session identifier.
            language: Requested sign language code ('ASL', etc.).
            raw_landmarks: Unmirrored 21x3 landmark coordinates, or None.
            handedness: Detected hand label ('Left' or 'Right').

        Returns:
            RecognitionResult with current frame state and suggestions.
        """
        with self.sessions.locked_session(session_id) as session:
            return self._process_frame_locked(
                session_id,
                session,
                language,
                raw_landmarks,
                handedness,
            )

    def _process_frame_locked(
        self,
        session_id: str,
        session: Session,
        language: str,
        raw_landmarks: Sequence[Sequence[float]] | None,
        handedness: str,
    ) -> RecognitionResult:
        """Process a frame while holding the session processing lock."""
        current_label = ""
        current_confidence = 0.0

        if raw_landmarks is not None and len(raw_landmarks) > 0:
            if len(raw_landmarks) != 21:
                raise InvalidLandmarkError(
                    f"Fingerspelling requires 21 landmarks, received {len(raw_landmarks)}"
                )

            # Canonicalise opposite hand coordinates to single standard convention
            canonical = canonicalise_landmarks(
                raw_landmarks, handedness, self._canonical_handedness
            )
            frame = HandFrame.from_raw(canonical)

            # Obtain language recognizer strategy
            recognizer = self._get_recognizer(session, language)
            gesture = recognizer.recognize(frame)

            if gesture is not None:
                current_label = gesture.label
                current_confidence = gesture.confidence
        else:
            gesture = None
            with session._recognizers_lock:
                existing_recognizer = session.recognizers.get(language)
            if existing_recognizer is not None and hasattr(existing_recognizer, "reset"):
                existing_recognizer.reset()

        # Pass through temporal stability filter
        state, confirmed_letter = session.stability_filter.update(gesture)

        # If a letter is stably confirmed, append to the active word
        if confirmed_letter is not None:
            session.word_builder.add_letter(confirmed_letter)
            if self.bus is not None:
                self.bus.publish(
                    "letter_confirmed",
                    {
                        "session_id": session_id,
                        "letter": confirmed_letter,
                        "current_word": session.word_builder.current_word,
                    },
                )

        current_word = session.word_builder.current_word
        sentence = session.word_builder.sentence
        suggestions = self.suggestions.suggest(current_word, limit=3)

        return RecognitionResult(
            label=current_label,
            confidence=round(current_confidence, 4),
            state=state.value,
            confirmed_letter=confirmed_letter,
            current_word=current_word,
            sentence=sentence,
            suggestions=suggestions,
        )

    def commit_word(self, session_id: str, word: str | None = None) -> RecognitionResult:
        """Commit the current spelled word (or suggested replacement) to the sentence."""
        with self.sessions.locked_session(session_id) as session:
            committed = session.word_builder.commit_word(word)

            if committed and self.bus is not None:
                self.bus.publish(
                    "word_committed",
                    {
                        "session_id": session_id,
                        "word": committed,
                        "language": "ASL",
                    },
                )

            return RecognitionResult(
                label="",
                confidence=0.0,
                state=session.stability_filter.state.value,
                confirmed_letter=None,
                current_word=session.word_builder.current_word,
                sentence=session.word_builder.sentence,
                suggestions=[],
            )

    def backspace(self, session_id: str, word: str | None = None) -> RecognitionResult:
        """Delete last character from the active word."""
        with self.sessions.locked_session(session_id) as session:
            if word is not None:
                session.word_builder._current_word = list(word.strip().upper())
            session.word_builder.backspace()
            current_word = session.word_builder.current_word

            return RecognitionResult(
                label="",
                confidence=0.0,
                state=session.stability_filter.state.value,
                confirmed_letter=None,
                current_word=current_word,
                sentence=session.word_builder.sentence,
                suggestions=self.suggestions.suggest(current_word, limit=3),
            )

    def reset_session(self, session_id: str) -> None:
        """Reset session stateful filter and words."""
        self.sessions.reset(session_id)
