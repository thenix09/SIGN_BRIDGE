"""American Sign Language (ASL) letter recognizer."""

from backend.app.recognition.factory import RecognizerFactory
from backend.app.recognition.landmark_recognizer import LandmarkRecognizer


class ASLRecognizer(LandmarkRecognizer):
    """ASL fingerspelling letter recognition using random forest landmark classification."""

    @property
    def language(self) -> str:
        """Return 'ASL' language identifier."""
        return "ASL"


# Auto-register ASL in factory
RecognizerFactory.register("ASL", ASLRecognizer)
