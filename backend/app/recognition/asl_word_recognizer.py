"""ASL-specific WordRecognizer implementation using OnnxWordRecognizer."""

from backend.app.recognition.factory import RecognizerFactory
from backend.app.recognition.onnx_word_recognizer import OnnxWordRecognizer


class ASLWordRecognizer(OnnxWordRecognizer):
    """ASL sequence WordRecognizer backed by an ONNX model."""

    def __init__(self, **kwargs) -> None:
        super().__init__(language="ASL", **kwargs)


# Register with factory
RecognizerFactory.register_word("ASL", ASLWordRecognizer)
