"""Abstract base classes and fakes for the gesture and sign recognition subsystems."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from backend.app.domain.gesture import Gesture

if TYPE_CHECKING:
    from backend.app.domain.hand_frame import HandFrame
    from backend.app.domain.sequence import LandmarkWindow


class SignRecognizer(ABC):
    """Abstract strategy for single-frame fingerspelling recognition."""

    @property
    @abstractmethod
    def language(self) -> str:
        """Return the ISO or standard language code (e.g. 'ASL', 'ISL')."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model weights, templates, or underlying recognizer resources."""
        pass

    @abstractmethod
    def recognize(self, frame: "HandFrame") -> Gesture | None:
        """Recognize a letter gesture from a single HandFrame.

        Args:
            frame: A validated and canonicalized HandFrame.

        Returns:
            A Gesture instance if recognized with confidence above threshold, else None.
        """
        pass


class WordRecognizer(ABC):
    """Abstract strategy for multi-frame isolated sign word recognition."""

    @property
    @abstractmethod
    def language(self) -> str:
        """Return the ISO or standard language code (e.g. 'ASL')."""
        pass

    @property
    @abstractmethod
    def vocabulary(self) -> tuple[str, ...]:
        """Return the closed vocabulary supported by this word recognizer."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load model weights or runtime session."""
        pass

    @abstractmethod
    def recognize(self, window: "LandmarkWindow", top_k: int = 3) -> list[Gesture]:
        """Recognize sign candidates from a LandmarkWindow of frames.

        Args:
            window: Preprocessed LandmarkWindow.
            top_k: Number of top candidate predictions to return.

        Returns:
            Top-k gestures ordered by confidence descending; never raises for low confidence.
        """
        pass


class FakeRecognizer(SignRecognizer):
    """Fake single-frame recognizer for testing and decoupled development."""

    def __init__(
        self, language: str = "ASL", default_label: str = "A", default_confidence: float = 0.95
    ) -> None:
        self._language = language
        self.default_label = default_label
        self.default_confidence = default_confidence
        self.is_loaded = False

    @property
    def language(self) -> str:
        """Return language code."""
        return self._language

    def load(self) -> None:
        """Simulate loading."""
        self.is_loaded = True

    def recognize(self, frame: Any) -> Gesture | None:
        """Return fake gesture prediction."""
        if not self.is_loaded:
            self.load()
        if self.default_confidence <= 0.0:
            return None
        return Gesture(label=self.default_label, confidence=self.default_confidence)


class FakeWordRecognizer(WordRecognizer):
    """Fake sequence word recognizer for testing and decoupled development."""

    def __init__(
        self,
        language: str = "ASL",
        vocabulary: tuple[str, ...] = ("hello", "thank you", "yes", "no"),
        default_predictions: list[tuple[str, float]] | None = None,
    ) -> None:
        self._language = language
        self._vocabulary = vocabulary
        self.default_predictions = default_predictions or [
            ("hello", 0.88),
            ("thank you", 0.08),
            ("yes", 0.04),
        ]
        self.is_loaded = False

    @property
    def language(self) -> str:
        """Return language code."""
        return self._language

    @property
    def vocabulary(self) -> tuple[str, ...]:
        """Return closed vocabulary."""
        return self._vocabulary

    def load(self) -> None:
        """Simulate loading."""
        self.is_loaded = True

    def recognize(self, window: Any, top_k: int = 3) -> list[Gesture]:
        """Return fake top-k word gesture predictions."""
        if not self.is_loaded:
            self.load()
        results = [
            Gesture(label=label, confidence=conf)
            for label, conf in self.default_predictions[:top_k]
        ]
        return results
