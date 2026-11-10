"""Indian Sign Language (ISL) recognizer stub proving multi-language extensibility."""

from backend.app.core.exceptions import ModelNotAvailableError
from backend.app.domain.gesture import Gesture
from backend.app.domain.hand_frame import HandFrame
from backend.app.recognition.factory import RecognizerFactory
from backend.app.recognition.landmark_recognizer import LandmarkRecognizer


class ISLRecognizer(LandmarkRecognizer):
    """Stub for Indian Sign Language demonstrating the open extension point."""

    @property
    def language(self) -> str:
        """Return 'ISL' language identifier."""
        return "ISL"

    def load(self) -> None:
        """Raise 503 error as ISL model weights are unavailable."""
        raise ModelNotAvailableError("Indian Sign Language (ISL) model is not currently available")

    def recognize(self, frame: HandFrame) -> Gesture | None:
        """Raise ModelNotAvailableError on recognition attempt."""
        raise ModelNotAvailableError("Indian Sign Language (ISL) model is not currently available")


# Auto-register ISL in factory
RecognizerFactory.register("ISL", ISLRecognizer)
