from collections import namedtuple
import pytest

from agents.dqn import DQN
from agents.rainbow import RainbowDQN
from core.env_details import EnvDetails
from core.parameters import (
    ModelParameters,
    DQNParameters,
    EnvParameters,
    Experience,
    RainbowDQNParameters,
    BufferParameters
)
from models._base import BaseModel
from models.dueling import CategoricalNoisyDueling

import torch
import torch.optim as optim
import torch.nn as nn


DQNExperience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


@pytest.fixture
def env_details() -> EnvDetails:
    return EnvDetails(EnvParameters('ALE/SpaceInvaders-v5', img_size=128, stack_size=4))


@pytest.fixture
def model_params(env_details) -> ModelParameters:
    network = BaseModel(input_shape=env_details.input_shape, n_actions=env_details.n_actions)
    return ModelParameters(
        network=network,
        optimizer=optim.Adam(params=network.parameters(), lr=1e-3, eps=1e-3),
        loss_metric=nn.MSELoss()
    )


@pytest.fixture
def dqn_params(env_details) -> DQNParameters:
    return DQNParameters(
        gamma=0.99,
        tau=1e-3,
        buffer_size=2,
        batch_size=1,
        update_steps=2,
    )


@pytest.fixture
def dqn_experience() -> DQNExperience:
    return DQNExperience(torch.randn((4, 128, 128)), 1, 1., torch.randn((4, 128, 128)), False)


@pytest.fixture
def rainbow_experience() -> Experience:
    return Experience(torch.randn((4, 128, 128)), 1, 1., False,
                      torch.randn((4, 128, 128)))


@pytest.fixture
def device() -> str:
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def dqn(env_details, model_params, dqn_params, device) -> DQN:
    return DQN(env_details, model_params, dqn_params, device=device, seed=1)


@pytest.fixture
def rdqn(env_details, device) -> RainbowDQN:
    params = RainbowDQNParameters(gamma=0.99, tau=1e3, update_steps=3, max_timesteps=100,
        n_atoms=10, v_min=-10, v_max=10, replay_period=10, n_steps=3, learn_frequency=3,
        clip_grad=0.5, reward_clip=0.1)
    buffer_params = BufferParameters(buffer_size=10, batch_size=1, priority_exponent=0.5,
        priority_weight=0.4, input_shape=env_details.input_shape)
    network = CategoricalNoisyDueling(input_shape=env_details.input_shape,
                                      n_actions=env_details.n_actions,
                                      n_atoms=10)
    model_params = ModelParameters(
        network=network,
        optimizer=optim.Adam(network.parameters(), lr=1e3, eps=1e3)
    )
    return RainbowDQN(env_details, model_params, params, buffer_params, device=device, seed=1)


def test_dqn_creation_invalid(env_details, model_params, device) -> None:
    assert DQN(env_details, model_params, DQNParameters(gamma=1), device=device, seed=1)


def test_dqn_act_valid(dqn, env_details) -> None:
    dqn.local_network = dqn.local_network.cpu()
    dqn.target_network = dqn.target_network.cpu()
    action = dqn.act(torch.randn(env_details.input_shape), 1.0)
    assert int(action) in range(env_details.n_actions)


def test_dqn_step_valid(dqn, dqn_experience) -> None:
    try:
        dqn.step(dqn_experience)
        assert True
    except ValueError:
        assert False


def test_dqn_learn_invalid(dqn, dqn_experience) -> None:
    try:
        dqn.learn(dqn_experience)
        assert False
    except RuntimeError:
        assert True


def test_dqn_learn_valid(dqn, dqn_experience) -> None:
    try:
        dqn.local_network = dqn.local_network.to('cpu')
        dqn.target_network = dqn.target_network.to('cpu')
        dqn.learn(dqn_experience)
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


def test_rainbow_act_valid(env_details, rdqn, device) -> None:
    action = rdqn.act(torch.randn(env_details.input_shape).unsqueeze(0).to(device))
    assert int(action) in range(env_details.n_actions)


def test_rainbow_learn_valid(rdqn) -> None:
    try:
        rdqn.learn()
        assert True
    except (RuntimeError, TypeError, ValueError):
        assert False


def test_rainbow_compute_double_probs_valid(env_details, rdqn, device) -> None:
    next_state = torch.randn(env_details.input_shape).unsqueeze(0).to(device)
    q_probs = rdqn.compute_double_q_probs(next_state)
    assert q_probs.shape == torch.rand((1, 10)).shape


def test_rainbow_train_valid(rdqn) -> None:
    try:
        rdqn.train(num_episodes=1)
        assert True
    except (RuntimeError, TypeError, ValueError):
        assert False
