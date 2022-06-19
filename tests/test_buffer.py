from collections import namedtuple
import pytest
import torch

from core.buffer import ReplayBuffer, RolloutBuffer


@pytest.fixture
def experience() -> namedtuple:
    Experience = namedtuple("Experience", field_names=['state', 'action', 'reward',
                                                'next_state', 'done'])
    return Experience(torch.ones((9, 9)), 1, 1., torch.ones((9, 9)), False)


@pytest.fixture
def rollout_buffer() -> RolloutBuffer:
    return RolloutBuffer(size=1, num_agents=1, env_input_shape=(4, 128, 128), action_shape=())


def test_replay_buffer_add_valid(experience) -> None:
    try:
        buffer = ReplayBuffer(buffer_size=10, batch_size=1, device='cpu', seed=1)
        buffer.add(experience)
        assert True
    except ValueError:
        assert False


def test_replay_buffer_sample_valid(experience) -> None:
    try:
        buffer = ReplayBuffer(buffer_size=10, batch_size=1, device='cpu', seed=1)
        buffer.add(experience)
        buffer.add(experience)
        buffer.sample()
        assert True
    except (ValueError, TypeError):
        assert False


def test_rollout_buffer_add_invalid_key(rollout_buffer) -> None:
    try:
        rollout_buffer.add(0, test='')
        assert False
    except ValueError:
        assert True


def test_rollout_buffer_add_invalid_value(rollout_buffer) -> None:
    try:
        rollout_buffer.add(0, actions=[1, 2, 1])
        assert False
    except (ValueError, TypeError):
        assert True


def test_rollout_buffer_add_valid(rollout_buffer) -> None:
    tensor = torch.ones((rollout_buffer.size, rollout_buffer.num_agents))
    rollout_buffer.add(0, actions=tensor)
    assert rollout_buffer.actions == tensor


def test_rollout_buffer_sample_valid(rollout_buffer) -> None:
    tensor = torch.zeros((rollout_buffer.size, rollout_buffer.num_agents))
    sample = rollout_buffer.sample(['actions'])
    assert sample.actions == tensor


def test_rollout_buffer_sample_batch_valid(rollout_buffer) -> None:
    tensor = torch.zeros((rollout_buffer.size, rollout_buffer.num_agents)).reshape(-1)
    batch = rollout_buffer.sample_batch(['actions'])
    assert batch.actions == tensor
