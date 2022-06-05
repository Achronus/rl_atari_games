from collections import namedtuple
import pytest

from agents.dqn import DQN
from core.env_details import EnvDetails
from core.parameters import ModelParameters, DQNParameters
from models.cnn import CNNModel

import torch
import torch.optim as optim
import torch.nn as nn


@pytest.fixture
def env_details() -> EnvDetails:
    return EnvDetails(gym_name='ALE/SpaceInvaders-v5', img_size=128, stack_size=4)


@pytest.fixture
def model_params(env_details: EnvDetails) -> ModelParameters:
    network = CNNModel(input_shape=env_details.input_shape, n_actions=env_details.n_actions, seed=1)
    return ModelParameters(
        network=network,
        optimizer=optim.Adam(params=network.parameters(), lr=1e-3, eps=1e-3),
        loss_metric=nn.MSELoss()
    )


@pytest.fixture
def dqn_params(env_details) -> DQNParameters:
    network = CNNModel(input_shape=env_details.input_shape, n_actions=env_details.n_actions, seed=1)
    return DQNParameters(
        gamma=0.99,
        tau=1e-3,
        buffer_size=10,
        batch_size=4,
        update_steps=4,
    )


@pytest.fixture
def experience() -> namedtuple:
    Experience = namedtuple("Experience", field_names=['state', 'action', 'reward',
                                                'next_state', 'done'])
    return Experience(torch.randn((4, 128, 128)), 1, 1.,
                      torch.randn((4, 128, 128)), False)


def test_dqn_creation_invalid(env_details, model_params) -> None:
    try:
        DQN(env_details, model_params, DQNParameters(gamma=1), seed=1)
        assert False
    except TypeError:
        assert True


def test_dqn_act_valid(env_details, model_params, dqn_params) -> None:
    dqn = DQN(env_details, model_params, dqn_params, seed=1)
    action = dqn.act(torch.randn(env_details.input_shape), 1.0)
    assert int(action) in range(env_details.n_actions)


def test_dqn_step_valid(env_details, model_params, dqn_params, experience) -> None:
    try:
        dqn = DQN(env_details, model_params, dqn_params, seed=1)
        dqn.step(experience)
        assert True
    except ValueError:
        assert False


def test_dqn_learn_invalid(env_details, model_params, dqn_params, experience) -> None:
    try:
        dqn = DQN(env_details, model_params, dqn_params, seed=1)
        dqn.learn(experience)
        assert False
    except RuntimeError:
        assert True


def test_dqn_learn_size_valid(env_details, model_params, dqn_params, experience) -> None:
    try:
        dqn = DQN(env_details, model_params, dqn_params, seed=1)
        dqn.local_network = dqn.local_network.to('cpu')
        dqn.target_network = dqn.target_network.to('cpu')
        dqn.learn(experience)
        assert False
    except RuntimeError:
        assert True


def test_dqn_log_data_invalid(env_details, model_params, dqn_params) -> None:
    try:
        dqn = DQN(env_details, model_params, dqn_params, seed=1)
        dqn.log_data(test='test')
        assert False
    except ValueError:
        assert True


def test_dqn_train_valid(env_details, model_params, dqn_params) -> None:
    try:
        dqn = DQN(env_details, model_params, dqn_params, seed=1)
        dqn.train(num_episodes=1, print_every=1)
        assert True
    except (ValueError, TypeError):
        assert False
