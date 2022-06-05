import _pickle
import os

from models.cnn import CNNModel
from utils.model_utils import load_model

import torch


def test_cnn_model_conv_size_valid() -> None:
    cnn = CNNModel(input_shape=(4, 128, 128), n_actions=6, seed=368)
    size = cnn.get_conv_size(input_shape=(4, 128, 128))
    assert size == 2048


def test_cnn_model_forward_valid() -> None:
    cnn = CNNModel(input_shape=(4, 128, 128), n_actions=6, seed=368)
    data = cnn.forward(torch.rand((128, 4, 128, 128)))
    assert data.shape == (128, 6)


def test_load_model_invalid_filename() -> None:
    try:
        dqn = load_model('test2', device='cpu', model_type='dqn')
        assert False
    except AssertionError:
        assert True


def test_load_model_invalid_model_type() -> None:
    filepath = 'saved_models/test.pt'
    try:
        if not os.path.exists(filepath):
            with open(filepath, 'w') as file:
                file.write('test.')
            file.close()

        model = load_model('test', device='cpu', model_type='a2c')
        assert False
    except ValueError:
        os.remove(filepath)
        assert True


def test_load_model_dqn_invalid_file_content() -> None:
    filepath = 'saved_models/test3.pt'
    try:
        if not os.path.exists(filepath):
            with open(filepath, 'w') as file:
                file.write('test.')
            file.close()

        model = load_model('test3', device='cpu', model_type='dqn')
        assert False
    except _pickle.UnpicklingError:
        os.remove(filepath)
        assert True


def test_load_model_dqn_valid() -> None:
    try:
        model = load_model('dqn_example', device='cpu', model_type='dqn')
        assert True
    except (ValueError, _pickle.UnpicklingError):
        assert False
