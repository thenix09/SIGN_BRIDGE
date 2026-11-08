import torch

from backend.app.recognition.word_model import WordSignModel


def test_word_sign_model_architecture_and_output_shape() -> None:
    model = WordSignModel()

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert model.gru.hidden_size == 128
    assert model.gru.bidirectional
    assert model.num_classes == 30
    assert trainable_parameters < 2_000_000
    assert model(torch.zeros(2, 32, 104)).shape == (2, 30)
