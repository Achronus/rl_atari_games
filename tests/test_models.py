from models.cnn import CNNModel

import torch


def test_cnn_model_conv_size_valid() -> None:
    cnn = CNNModel(input_shape=(4, 128, 128), n_actions=6, seed=368)
    size = cnn.get_conv_size(input_shape=(4, 128, 128))
    assert size == 2048


def test_cnn_model_forward_valid() -> None:
    cnn = CNNModel(input_shape=(4, 128, 128), n_actions=6, seed=368)
    data = cnn.forward(torch.rand((128, 4, 128, 128)))
    assert data.shape == (128, 6)
