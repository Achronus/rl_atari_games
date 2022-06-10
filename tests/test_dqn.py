from collections import namedtuple
import pytest

from agents.dqn import DQN
from core.env_details import EnvDetails
from core.parameters import ModelParameters, DQNParameters, EnvParameters
from models.cnn import CNNModel

import torch
import torch.optim as optim
import torch.nn as nn


@pytest.fixture
def env_details() -> EnvDetails:
    return EnvDetails(EnvParameters('ALE/SpaceInvaders-v5', img_size=128, stack_size=4))


@pytest.fixture
def model_params(env_details: EnvDetails) -> ModelParameters:
    network = CNNModel(input_shape=env_details.input_shape, n_actions=env_details.n_actions)
    return ModelParameters(
        network=network,
        optimizer=optim.Adam(params=network.parameters(), lr=1e-3, eps=1e-3),
        loss_metric=nn.MSELoss()
    )


@pytest.fixture
def dqn_params(env_details) -> DQNParameters:
    network = CNNModel(input_shape=env_details.input_shape, n_actions=env_details.n_actions)
    return DQNParameters(
        gamma=0.99,
        tau=1e-3,
        buffer_size=2,
        batch_size=1,
        update_steps=2,
    )


@pytest.fixture
def experience() -> namedtuple:
    Experience = namedtuple("Experience", field_names=['state', 'action', 'reward',
                                                'next_state', 'done'])
    return Experience(torch.randn((4, 128, 128)), 1, 1.,
                      torch.randn((4, 128, 128)), False)


@pytest.fixture
def dqn(env_details, model_params, dqn_params) -> DQN:
    return DQN(env_details, model_params, dqn_params, seed=1)


def test_dqn_creation_invalid(env_details, model_params) -> None:
    try:
        DQN(env_details, model_params, DQNParameters(gamma=1), seed=1)
        assert False
    except TypeError:
        assert True


def test_dqn_act_valid(dqn, env_details) -> None:
    dqn.local_network = dqn.local_network.cpu()
    dqn.target_network = dqn.target_network.cpu()
    action = dqn.act(torch.randn(env_details.input_shape), 1.0)
    assert int(action) in range(env_details.n_actions)


def test_dqn_step_valid(dqn, experience) -> None:
    try:
        dqn.step(experience)
        assert True
    except ValueError:
        assert False


def test_dqn_learn_invalid(dqn, experience) -> None:
    try:
        dqn.learn(experience)
        assert False
    except RuntimeError:
        assert True


def test_dqn_learn_size_valid(dqn, experience) -> None:
    try:
        dqn.local_network = dqn.local_network.to('cpu')
        dqn.target_network = dqn.target_network.to('cpu')
        dqn.learn(experience)
        assert False
    except RuntimeError:
        assert True


def test_dqn_log_data_invalid(dqn) -> None:
    try:
        dqn.log_data(test='test')
        assert False
    except ValueError:
        assert True


def test_dqn_log_data_valid(dqn) -> None:
    try:
        dqn.log_data(actions=[1, 2, 3])
        assert True
    except (ValueError, TypeError):
        assert False


def test_dqn_initial_output_invalid(dqn) -> None:
    try:
        assert dqn._initial_output(1, extra_info=2)
        assert False
    except AssertionError:
        assert True


def test_dqn_train_valid(dqn) -> None:
    try:
        dqn.train(num_episodes=1, print_every=1)
        assert True
    except (ValueError, TypeError):
        assert False
