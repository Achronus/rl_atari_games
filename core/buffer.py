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

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.memory)
