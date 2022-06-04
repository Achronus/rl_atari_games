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
def dqn_params() -> DQNParameters:
    network = CNNModel(input_shape=env_details.input_shape, n_actions=env_details.n_actions, seed=1)
    return DQNParameters(
        gamma=0.99,
        tau=1e-3,
        buffer_size=100,
        batch_size=32,
        update_steps=4,
        target_network=network
    )


@pytest.fixture
def dqn(env_details: EnvDetails, model_params: ModelParameters, dqn_params: DQNParameters) -> DQN:
    return DQN(env_details, model_params, dqn_params, seed=1)


def dqn_act_valid() -> None:
    action = dqn.act(torch.randn(env_details.input_shape), 1.0)
    assert int(action)


def dqn_load_model_invalid() -> None:
    try:
        dqn.load_model('test', device='cuda:0')
        assert False
    except AssertionError:
        assert True
