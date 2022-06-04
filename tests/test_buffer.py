from collections import namedtuple
import pytest
import torch

from core.buffer import ReplayBuffer
from core.env_details import EnvDetails


@pytest.fixture
def env_details() -> EnvDetails:
    return EnvDetails(gym_name='ALE/SpaceInvaders-v5', img_size=128, stack_size=4)


@pytest.fixture
def experience() -> namedtuple:
    Experience = namedtuple("Experience", field_names=['state', 'action', 'reward',
                                                'next_state', 'done'])
    return Experience(torch.ones((9, 9)), 1, 1., torch.ones((9, 9)), False)


def test_buffer_add_valid(env_details, experience) -> None:
    try:
        buffer = ReplayBuffer(env_details, buffer_size=10, batch_size=1,
                              device='cpu', seed=1)
        buffer.add(experience)
        assert True
    except ValueError:
        assert False


def test_buffer_stack_valid(env_details, experience) -> None:
    try:
        buffer = ReplayBuffer(env_details, buffer_size=10, batch_size=1,
                              device='cpu', seed=1)
        buffer.add(experience)
        buffer.add(experience)
        buffer.sample()
        assert True
    except (ValueError, TypeError):
        assert False
