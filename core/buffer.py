import random
from collections import deque, namedtuple

import numpy as np
import torch

from core.env_details import EnvDetails

Experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    A basic representation of an experience replay buffer.

    Parameters:
        env_details (EnvDetails) - a class containing parameters for the environment
        buffer_size (int) - size of the memory buffer
        batch_size (int) - size of each training batch
        device (str) - device name for data calculations (CUDA GPU or CPU)
        seed (int) - a random number for recreating results
    """
    def __init__(self, env_details: EnvDetails, buffer_size: int, batch_size: int,
                 device: str, seed: int) -> None:
        self.action_size = env_details.n_actions
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
