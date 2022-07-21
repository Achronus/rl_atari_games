import random
from typing import Union
from collections import deque, namedtuple

from core.parameters import BufferParameters, Experience

import numpy as np
import torch

DQNExperience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    A basic representation of an experience replay buffer.

    :param buffer_size (int) - size of the memory buffer
    :param batch_size (int) - size of each training batch
    :param device (str) - device name for data calculations (CUDA GPU or CPU)
    :param seed (int) - a random number for recreating results
    """

    def __init__(self, buffer_size: int, batch_size: int, device: str, seed: int) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        random.seed(seed)

        self.memory = deque(maxlen=self.buffer_size)

    def add(self, experience: DQNExperience) -> None:
        """Add a tuple of experience to the buffer memory."""
        self.memory.append(experience)

    def sample(self) -> tuple:
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

    :param capacity (int) - size of tree memory (number of experiences to store)
    :param device (str) - CUDA device to store the data on (CPU or GPU)
    :param n_steps (int) - number of steps to use for multi-step (N-step) learning
    :param input_shape (tuple) - the environment state's input shape (width, height)
    """
    def __init__(self, capacity: int, device: str, n_steps: int, input_shape: tuple) -> None:
        self.position = 0  # Pointer
        self.data_count = 0  # Track data entries (not placeholders)
        self.capacity = capacity
        self.n_steps = n_steps

        self.num_nodes = 2 * capacity - 1  # All nodes
        self.priorities = torch.zeros(self.num_nodes).to(device)  # All tree nodes
        self.max_priority = 1  # initial

        self.data = np.zeros((capacity,), dtype=object)  # Data storage (leaf nodes)

        placeholder_transition = Experience(
            state=torch.zeros(input_shape[1:]),
            action=0,
            reward=0.,
            done=False
        )
        # Fill data with placeholder values
        for idx in range(self.data.shape[0]):
            self.data[idx] = placeholder_transition

    def add(self, priority: int, experience: Experience) -> None:
        """Add an experience to the tree with a given priority."""
        self.data[self.position] = experience

        # Update the tree
        tree_idx = self.position + self.capacity - 1
        self.__update_node(tree_idx, priority)

        # Update pointer and maximum priority
        self.position = (self.position + 1) % self.capacity
        self.max_priority = max(priority, self.max_priority)

        # Increment data count
        if self.data_count != self.capacity:
            self.data_count += 1

    def __update_node(self, idx: int, priority: int) -> None:
        """Adds priority value to respective node in the tree."""
        self.priorities[idx] = priority
        self.__update_tree(idx)

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

        # Manage batch items that have finished descending tree before others (index overshooting)
        finished_bools = torch.gt(child_indices, self.num_nodes)  # check if > num nodes
        if torch.any(finished_bools):
            # Get finished indices
            _, finished_indices = torch.where(finished_bools)
            left_size = np.count_nonzero(finished_bools[0])  # For slicing

            left = finished_indices[left_size:]
            right = finished_indices[:left_size]

            # Replace child indices with old ones
            child_indices[0][left] = indices[left]
            child_indices[1][right] = indices[right]

        # Compare priorities to left node
        # Moving down tree based on comparison -> 0 = left node, 1 = right node
        left_child_values = self.priorities[child_indices[0]]
        comparison_bools = torch.gt(priorities, left_child_values).to(torch.long)  # priority > left_child = 1, else 0

        # Set next indices
        next_indices = child_indices[comparison_bools, torch.arange(indices.size()[0])]

        # Reduce priorities to move down tree
        next_priorities = priorities - (comparison_bools * left_child_values)

        # Recursively run until get leaf indices
        return self.__tree_search(next_indices, next_priorities)

    def sample(self, priorities: torch.Tensor) -> dict:
        """Samples an array of values from the tree.
        Returns the priority values, data indices, and tree indices."""
        init_indices = torch.zeros(priorities.shape, dtype=torch.long)  # Start for recursive tree search
        indices = self.__tree_search(init_indices, priorities)
        data_indices = indices % self.capacity

        samples = dict(
            priorities=self.priorities[indices].detach(),
            transition_indices=data_indices,
            priority_indices=indices
        )
        return samples

    def get_data(self, indices: torch.Tensor) -> np.ndarray:
        """Gets data from memory given a tensor of indices."""
        return self.data[indices % self.capacity]

    def total_priority(self) -> float:
        """Returns the total priority (root node value)."""
        return self.priorities[0].detach().item()

    def __len__(self) -> int:
        """Returns the current amount of data in memory."""
        return self.data_count


class PrioritizedReplayBuffer:
    """
    A prioritized experience replay buffer. As illustrated in the Prioritized Experience Replay paper:
    https://arxiv.org/pdf/1511.05952.pdf.

    :param params (BufferParameters) - a class containing parameters for the buffer
    :param device (str) - CUDA device name for data calculations (GPU or CPU)
    :param n_steps (int) - Number of steps for multi-step learning
    :param stack_size (int) - number of state frames to stack together
    """
    def __init__(self, params: BufferParameters, n_steps: int, stack_size: int, device: str, logger) -> None:
        self.params = params
        self.capacity = params.buffer_size
        self.batch_size = params.batch_size
        self.device = device
        self.n_steps = n_steps
        self.history_length = stack_size

        self.logger = logger

        self.priority_weight = params.priority_weight
        self.priority_exponent = params.priority_exponent

        self.memory = SumTree(self.capacity, device, n_steps, self.params.input_shape)

    def add(self, experience: Experience) -> None:
        """Add a single experience to the buffer memory with maximum priority."""
        self.memory.add(self.memory.max_priority, experience)

    def sample(self) -> dict:
        """Samples a batch of experiences from memory."""
        # Compute uniform priorities
        priority_total = self.total_priority()
        segment_size = priority_total / self.batch_size
        segments = torch.arange(self.batch_size) * segment_size  # Uniform list from 0 to priority_total
        priorities = torch.zeros(self.batch_size).uniform_(to=segment_size) + segments

        # Create N-step matrix and get experiences
        exp_info = self.memory.sample(priorities.to(self.device))
        n_step_indices = self.create_n_step_matrix(exp_info['transition_indices'])
        n_step_experiences = self.memory.get_data(n_step_indices)

        # Obtain a sample of experiences
        exp_dict = self.__compile_experiences(n_step_experiences)

        samples = {**exp_info, **exp_dict}  # Merge dictionaries
        return samples

    def create_n_step_matrix(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Creates an N-step matrix of transition indices using a given set of indices. Accounts for stacked frames
        in the environment.
        The matrix contains rows of [batch_size, range_indices], where range_indices is equivalent to:
        range(-self.history_length + 1, n_steps + 1).

        Range indices example with 2 batches, 4 frames stacked and 3 N-steps:
        [[-3, -2, -1, 0, 1, 2, 3]
         [-2, -1, 0, 1, 2, 3, 4]]
        """
        range_indices = torch.arange(-self.history_length + 1, self.n_steps + 1)
        return range_indices + np.expand_dims(indices, axis=1)

    def __compile_experiences(self, experiences: np.ndarray) -> dict:
        """Takes an array of dataclass objects and splits the attributes into respective tensors."""
        exp_tensors = self.__init_exp_tensors()

        for idx, row in enumerate(experiences):
            row_states = self.__stack_and_tensor(self.__get_experience('state', row))
            row_actions = self.__vstack_and_tensor(self.__get_experience('action', row)).to(torch.int64)
            row_rewards = self.__vstack_and_tensor(self.__get_experience('reward', row))
            row_dones = self.__vstack_and_tensor(self.__get_experience('done', row))

            exp_tensors['states'][idx] = row_states
            exp_tensors['actions'][idx] = row_actions.squeeze(1)
            exp_tensors['rewards'][idx] = row_rewards.squeeze(1)
            exp_tensors['dones'][idx] = row_dones.squeeze(1)

        # Set data based on N-steps
        return dict(
            states=exp_tensors['states'][:, :self.history_length],  # first few states
            actions=exp_tensors['actions'][:, self.history_length - 1].to(torch.long),  # last actions (before N-step)
            rewards=exp_tensors['rewards'][:, self.history_length - 1: -1],  # N-step rewards
            next_states=exp_tensors['states'][:, self.n_steps:],  # N-step states
            dones=exp_tensors['dones'][:, -1].unsqueeze(1)  # N-step mask
        )

    def __init_exp_tensors(self) -> dict:
        """Creates empty torch tensors based on the batch size and placeholder experience."""
        batch_shape = (self.batch_size, self.history_length + self.n_steps)
        state_batch_shape = (batch_shape + self.params.input_shape[1:])

        rewards = torch.zeros(batch_shape)
        states = torch.zeros(state_batch_shape)
        actions = torch.zeros(batch_shape)
        dones = torch.zeros(batch_shape)

        return dict(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones
        )

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
    def __get_experience(item: str, experiences: np.ndarray) -> list:
        """Gets a single component of samples from a sample of experiences."""
        return [getattr(exp, item) for exp in experiences if exp is not None]

    def importance_sampling(self, priorities: torch.Tensor) -> torch.Tensor:
        """Computes the importance sampling weights from a given set of priorities."""
        weights = (len(self.memory) * priorities) ** -self.priority_weight
        weights = (weights / weights.max()).to(self.device)
        return weights

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor) -> None:
        """Updates the priorities at a given set of indices."""
        priorities = torch.pow(priorities, self.priority_exponent)  # Increment priorities
        self.memory.update(indices, priorities)

    def total_priority(self) -> float:
        """Returns the total priority stored in memory (root node value)."""
        return self.memory.total_priority()

    def __len__(self) -> int:
        """Returns the current size of the data in the buffer."""
        return len(self.memory)


class RolloutBuffer:
    """
    A rollout buffer for storing agent experiences and other useful metrics.

    :param size (int) - number of items to store in the buffer
    """

    def __init__(self, size: int, num_envs: int, env_input_shape: tuple, action_shape: tuple) -> None:
        self.keys = ['states', 'actions', 'rewards', 'dones', 'log_probs', 'state_values', 'next_states']
        self.size = size
        self.num_envs = num_envs
        self.env_input_shape = env_input_shape
        self.action_shape = action_shape
        self.states = None
        self.next_states = None
        self.actions = None

        self.reset()

    def add(self, step: int, **kwargs) -> None:
        """Adds an item to the buffer."""
        for key, val in kwargs.items():
            if key not in self.keys:
                raise ValueError(f"Invalid key! Available keys: '{self.keys}'.")

            getattr(self, key)[step] = val

    def reset(self) -> None:
        """Resets keys to placeholder values."""
        self.states = torch.zeros((self.size, self.num_envs) + self.env_input_shape)
        self.next_states = torch.zeros((self.size, self.num_envs) + self.env_input_shape)
        self.actions = torch.zeros((self.size, self.num_envs) + self.action_shape)

        predefined_keys = ['states', 'next_states', 'actions']
        for key in self.keys:
            if key not in predefined_keys:
                setattr(self, key, torch.zeros((self.size, self.num_envs)))

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
            if key == 'states' or key == 'next_states':
                samples = getattr(self, key).reshape((-1,) + self.env_input_shape)
            elif key == 'actions':
                samples = getattr(self, key).reshape((-1,) + self.action_shape)
            else:
                samples = getattr(self, key).reshape(-1)
            data.append(samples)

        Batch = namedtuple('Batch', keys)
        return Batch(*data)
