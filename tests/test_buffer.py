from collections import namedtuple

import numpy as np
import pytest
import torch

from core.buffer import ReplayBuffer, RolloutBuffer, SumTree, PrioritizedReplayBuffer
from core.parameters import Experience, BufferParameters
from utils.logger import RDQNLogger


@pytest.fixture
def experience() -> namedtuple:
    return Experience(torch.ones((9, 9)), 1, 1., False, torch.ones((9, 9)))


@pytest.fixture
def rollout_buffer() -> RolloutBuffer:
    return RolloutBuffer(size=1, num_agents=1, env_input_shape=(4, 128, 128), action_shape=())


@pytest.fixture
def sum_tree() -> SumTree:
    return SumTree(10, 'cpu', 3, (10, 10))


@pytest.fixture
def priority_buffer() -> PrioritizedReplayBuffer:
    params = BufferParameters(10, 1, 0.1, 0.5, 3, (10, 10))
    return PrioritizedReplayBuffer(params, 4, 'cpu', RDQNLogger())


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


def test_sum_tree_add_valid(experience, sum_tree) -> None:
    try:
        sum_tree.add(1, experience)
        assert True
    except (ValueError, RuntimeError):
        assert False


def test_sum_tree_update_valid(experience, sum_tree) -> None:
    try:
        sum_tree.add(1, experience)
        sum_tree.update(torch.LongTensor((1,)), torch.Tensor((3,)))
        assert True
    except (ValueError, RuntimeError):
        assert False


def test_sum_tree_total_priority_valid(sum_tree) -> None:
    out = sum_tree.sample(torch.Tensor((1,)))
    out_type = isinstance(out, dict)
    out_items = []
    for val in out.values():
        out_items.append(isinstance(val, torch.Tensor))
    assert all([out_type, out_items])


def test_sum_tree_get_data_valid(sum_tree) -> None:
    output = sum_tree.get_data(torch.LongTensor((1, 2)))
    assert isinstance(output, np.ndarray)


def test_sum_tree_len_valid(sum_tree, experience) -> None:
    len_type = isinstance(len(sum_tree), int)
    val1_type = len(sum_tree) == 0
    sum_tree.add(1, experience)
    val2_type = len(sum_tree) == 1
    assert all([len_type, val1_type, val2_type])


def test_priority_buffer_add_valid(experience, priority_buffer) -> None:
    try:
        priority_buffer.add(experience)
        assert True
    except (RuntimeError, ValueError):
        assert False


def test_priority_buffer_sample_valid(priority_buffer) -> None:
    try:
        output = priority_buffer.sample()
        assert True
    except (RuntimeError, ValueError):
        assert False


def test_priority_buffer_create_matrix_valid(priority_buffer) -> None:
    in_tensor = torch.LongTensor((1, 2, 3))
    output = priority_buffer.create_n_step_matrix(in_tensor)
    valid_matrix = torch.LongTensor([[-2, -1, 0, 1, 2, 3, 4],
                                     [-1, 0, 1, 2, 3, 4, 5],
                                     [0, 1, 2, 3, 4, 5, 6]])
    assert torch.all(output == valid_matrix)


def test_priority_buffer_update_priorities_valid(priority_buffer) -> None:
    try:
        priority_buffer.update_priorities(torch.LongTensor((1, 2)), torch.Tensor((2, 3)))
        assert True
    except (RuntimeError, ValueError):
        assert False



