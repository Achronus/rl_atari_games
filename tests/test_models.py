import _pickle
import os

import pytest

from models.actor_critic import ActorCritic
from models.cnn import CNNModel
from models.dueling import CategoricalNoisyDueling
from models.linear import NoisyLinear
from utils.dataloader import DataLoader
from utils.model_utils import load_model

import torch


@pytest.fixture
def n_actions() -> int:
    return 6


@pytest.fixture
def input_shape() -> tuple[int, int, int]:
    return 4, 128, 128


def test_cnn_model_conv_size_valid(n_actions, input_shape) -> None:
    cnn = CNNModel(input_shape=input_shape, n_actions=n_actions)
    size = cnn.get_conv_size(input_shape=input_shape)
    assert size == 9216


def test_cnn_model_forward_valid(n_actions, input_shape) -> None:
    cnn = CNNModel(input_shape=input_shape, n_actions=n_actions)
    data = cnn.forward(torch.rand((128,) + input_shape))
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
    except (ValueError, _pickle.UnpicklingError):
        os.remove(filepath)
        assert True


def test_load_model_rqdn_invalid_model_file() -> None:
    try:
        model = load_model('dqn_example', device='cpu', model_type='rainbow')
        assert False
    except AttributeError:
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


def test_actor_critic_model_output_valid(n_actions, input_shape) -> None:
    batch_size = 10
    ac = ActorCritic(input_shape=input_shape, n_actions=n_actions)

    action_probs, state_values = ac.forward(torch.rand((batch_size,) + input_shape))
    ap_valid = action_probs.shape == torch.zeros((batch_size, n_actions)).shape
    svs_valid = state_values.shape == torch.zeros((batch_size, 1)).shape
    assert all([ap_valid, svs_valid])


def test_noisy_linear_forward_valid() -> None:
    torch_in = torch.rand((128,))
    noisy_layer = NoisyLinear(128, 9)
    output = noisy_layer.forward(torch_in)
    assert all([output.shape, torch.ones(128, 9).shape])


def test_noisy_dueling_network_valid(n_actions, input_shape) -> None:
    batch_size = 10
    n_atoms = 5
    dueling = CategoricalNoisyDueling(input_shape=input_shape, n_actions=n_actions,
                                      n_atoms=n_atoms)

    q_values = dueling.forward(torch.rand((batch_size,) + input_shape))
    assert all([q_values.shape, torch.rand((batch_size, n_actions, n_atoms)).shape])
