import random
from typing import Union
from collections import deque, namedtuple

from core.parameters import BufferParameters, Experience

import numpy as np
import torch


class ReplayBuffer:
    """
    A basic representation of an experience replay buffer.

    Parameters:
        buffer_size (int) - size of the memory buffer
        batch_size (int) - size of each training batch
        device (str) - device name for data calculations (CUDA GPU or CPU)
        seed (int) - a random number for recreating results
    """
    def __init__(self, buffer_size: int, batch_size: int, device: str, seed: int) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        random.seed(seed)

        self.memory = deque(maxlen=self.buffer_size)

    def add(self, experience: Experience) -> None:
        """Add a tuple of experience to the buffer memory."""
        self.memory.append(experience)

    def sample(self) -> tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # Sample every component individually
        states = self.__stack_and_tensor(self.__get_experience('state', experiences))
        actions = self.__vstack_and_tensor(self.__get_experience('action', experiences)).to(torch.int64)
        rewards = self.__vstack_and_tensor(self.__get_experience('reward', experiences))
        next_states = self.__stack_and_tensor(self.__get_experience('next_state', experiences))
        dones = self.__vstack_and_tensor(self.__get_experience('done', experiences))
        return states, actions, rewards, next_states, dones

    def __stack_and_tensor(self, items: list) -> torch.Tensor:
        """
        Stacks a list of values into a numpy array and then converts it to a PyTorch Tensor.
        Items are placed on their respective device (e.g., GPU is available).
        """
        return torch.from_numpy(np.stack(items)).to(self.device).to(torch.float32)

    def __vstack_and_tensor(self, items: list) -> torch.Tensor:
        """
        Vertically stacks a list of values into a numpy array and then converts it to a PyTorch Tensor.
        Items are placed on their respective device (e.g., GPU is available).
        """
        return torch.from_numpy(np.vstack(items)).to(self.device).to(torch.float32)

    @staticmethod
    def __get_experience(item: str, experiences: list) -> list:
        """Gets a single component of samples from a sample of experiences."""
        return [getattr(exp, item) for exp in experiences if exp is not None]

    def __len__(self) -> int:
        """Returns the current size of the buffer."""
        return len(self.memory)


class SumTree:
    """A binary tree data structure for storing replay buffer transitions. Each leaf contains the
    priority score for a single experience.

    Parameters:
        capacity (int) - size of tree memory (number of experiences to store)
        device (str) - CUDA device to store the data on (CPU or GPU)
    """
    def __init__(self, capacity: int, device: str) -> None:
        self.position = 0  # Pointer
        self.capacity = capacity

        self.num_nodes = 2 * capacity - 1  # Internal nodes
        self.priorities = torch.zeros(self.num_nodes).to(device)  # All tree nodes
        self.data = np.empty((capacity,), dtype=object)  # Data storage (leaf nodes)
        self.max_priority = 1  # initial

    def add(self, priority: int, experience: Experience) -> None:
        """Add an experience to the tree with a given priority."""
        # Add the experience to storage
        self.data[self.position] = experience

        # Update the tree
        tree_idx = self.position + self.capacity - 1
        self.__update_node(tree_idx, priority)

        # Update pointer
        self.position = (self.position + 1) % self.capacity

        # Reset index if max capacity (overwrite existing)
        if self.position >= self.capacity:
            self.position = 0

        # Update maximum priority
        self.max_priority = max(priority, self.max_priority)

    def __update_node(self, idx: int, priority: int) -> None:
        """Adds priority value to respective node in the tree."""
        self.priorities[idx] = priority
        self.__update_tree(idx)
        self.max_priority = max(priority, self.max_priority)  # Update max priority

    def __update_tree(self, idx: Union[int, torch.Tensor]) -> None:
        """Recursively propagates the new priority value(s) up the tree."""
        # Handle single
        if isinstance(idx, int):
            self.__propagate_single(idx)
        # Handle tensor
        elif isinstance(idx, torch.Tensor):
            self.__propagate_multi(idx)

    def __propagate_single(self, idx: int) -> None:
        """Propagates a single priority value up the tree."""
        # Update parent node
        parent_idx = (idx - 1) // 2
        left_idx = 2 * parent_idx + 1
        right_idx = left_idx + 1
        self.priorities[parent_idx] = self.priorities[left_idx] + self.priorities[right_idx]

        # Stop recursion when updated root
        if parent_idx != 0:
            self.__update_tree(parent_idx)

    def __propagate_multi(self, indices: torch.Tensor) -> None:
        """Propagates an array of priority values up the tree."""
        # Get parent nodes
        parent_indices = torch.div(indices - 1, 2, rounding_mode='floor')
        unique_parents = torch.unique(parent_indices)  # indices

        # Update parent priorities (sum of its child nodes)
        child_indices = unique_parents * 2 + np.expand_dims([1, 2], axis=1)  # left and right
        self.priorities[unique_parents] = torch.sum(self.priorities[child_indices], dim=0)

        # Repeat until root has been updated
        if parent_indices[0] != 0:
            self.__update_tree(parent_indices)

    def update(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        """Update one or more indices with new priorities."""
        # Set new values and update tree
        self.priorities[indices] = priorities
        self.__update_tree(indices)

        # Update maximum priority
        current_max = torch.max(priorities).item()
        self.max_priority = max(current_max, self.max_priority)

    def __tree_search(self, indices: torch.Tensor, priorities: torch.Tensor) -> torch.Tensor:
        """Obtains the indices of the given priorities by searching the tree."""
        # Get left and right node indices - row[0] = left, row[1] = right
        child_indices = (indices * 2 + np.expand_dims([1, 2], axis=1))  # shape -> [2, batch_size]

        # If indices are leaf nodes, return them
        if child_indices[0, 0] >= self.priorities.shape[0]:
            return indices  # shape -> [batch_size,]

        # Compare priorities to left node
        left_child_values = self.priorities[child_indices[0]]
        comparison_bools = torch.gt(priorities, left_child_values).to(torch.long)  # priority > left_child = 1, else 0

        # Move down tree based on comparison -> 0 = left node, 1 = right node
        next_indices = child_indices[comparison_bools, torch.arange(indices.size()[0])]

        # Reduce priorities to move down tree
        next_priorities = priorities - (comparison_bools * left_child_values)

        # Recursively run until get leaf indices
        return self.__tree_search(next_indices, next_priorities)

    def sample(self, priorities: torch.Tensor) -> dict:
        """Samples an array of values from the tree.
        Returns the priority values, data index, and tree indices."""
        init_indices = torch.zeros(priorities.shape, dtype=torch.long)  # Start for recursive tree search
        indices = self.__tree_search(init_indices, priorities)
        data_idx = indices - self.num_nodes
        return {
            'priorities': self.priorities[indices],
            'transitions': self.data[data_idx],
            'transition_indices': data_idx,
            'priority_indices': indices
        }

    def total_priority(self) -> float:
        """Returns the total priority (root node value)."""
        return self.priorities[0].cpu().item()

    def __len__(self) -> int:
        return np.count_nonzero(self.data)


class PrioritizedReplayBuffer:
    """
    A prioritized experience replay buffer. As illustrated in the Prioritized Experience Replay paper:
    https://arxiv.org/pdf/1511.05952.pdf.

    Parameters:
        params (BufferParameters) - a class containing parameters for the buffer
        device (str) - CUDA device name for data calculations (GPU or CPU)
    """
    def __init__(self, params: BufferParameters, device: str) -> None:
        self.params = params
        self.capacity = params.buffer_size
        self.batch_size = params.batch_size
        self.device = device

        self.max_priority = 1
        self.priority_weight = params.priority_weight
        self.priority_weight_increase: float = 0.

        self.memory = SumTree(self.capacity, device)

    def add(self, priority: int, experience: Experience) -> None:
        """Add a single experience to the buffer memory."""
        self.memory.add(priority, experience)

    def init_priority_weight_increase(self, num_episodes: int) -> None:
        """Initializes the priority weight increase."""
        self.priority_weight_increase = (1 - self.priority_weight) / (num_episodes - self.params.replay_period)

    def sample(self) -> tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences from memory."""
        # Anneal importance sampling weight (β to 1)
        self.priority_weight = min(self.priority_weight + self.priority_weight_increase, 1)
        pass

    def __len__(self) -> int:
        """Returns the current size of the buffer."""
        return len(self.memory)


class RolloutBuffer:
    """
    A rollout buffer for storing agent experiences and other useful metrics.

    Parameters:
        size (int) - number of items to store in the buffer
    """
    def __init__(self, size: int, num_agents: int, env_input_shape: tuple[int, ...],
                 action_shape: tuple[int, ...]) -> None:
        self.keys = ['states', 'actions', 'rewards', 'dones', 'log_probs', 'state_values']
        self.size = size
        self.num_agents = num_agents
        self.env_input_shape = env_input_shape
        self.action_shape = action_shape
        self.states = None
        self.actions = None

        self.reset()

    def add(self, step: int, **kwargs) -> None:
        """Adds an item to the buffer."""
        for key, val in kwargs.items():
            if key not in self.keys:
                raise ValueError(f"Invalid key! Available keys: '{self.keys}'.")

            if key == 'next_state':
                getattr(self, key)[0] = val
            else:
                getattr(self, key)[step] = val

    def reset(self) -> None:
        """Resets keys to placeholder values."""
        self.states = torch.zeros((self.size, self.num_agents) + self.env_input_shape)
        self.actions = torch.zeros((self.size, self.num_agents) + self.action_shape)

        predefined_keys = ['states', 'actions']
        for key in self.keys:
            if key not in predefined_keys:
                setattr(self, key, torch.zeros((self.size, self.num_agents)))

    def sample(self, keys: list) -> namedtuple:
        """Samples data from the buffer based on the provided keys."""
        data = [getattr(self, key) for key in keys]
        Sample = namedtuple('Sample', keys)
        return Sample(*data)

    def sample_batch(self, keys: list) -> namedtuple:
        """
        Samples a batch of data from the buffer based on the provided keys
        but converts them into mini-batches before returning them.
        """
        data = []
        for key in keys:
            if key == 'states':
                samples = getattr(self, key).reshape((-1,) + self.env_input_shape)
            elif key == 'actions':
                samples = getattr(self, key).reshape((-1,) + self.action_shape)
            else:
                samples = getattr(self, key).reshape(-1)
            data.append(samples)

        Batch = namedtuple('Batch', keys)
        return Batch(*data)
