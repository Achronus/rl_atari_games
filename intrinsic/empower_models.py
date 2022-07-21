"""
A family of neural networks for the empowerment intrinsic motivation method. Based on the Empowerment-driven Exploration
using Mutual Information Estimation paper: https://arxiv.org/abs/1810.05533.
"""
from models._base import BaseModel
from models.linear import NoisyLinear

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(BaseModel):
    """An encoder network that reduces a states dimensionality and provides an output equivalent
    to a single state space dimension. For example, (32, 4, 84, 84) -> (32, 84).

    Uses the same layers from the BaseModel class with a different output layer.

    :param input_shape (tuple[int]) - image input dimensions (including batch size at first dimension)
    :param n_actions (int) - number of possible actions in the environment
    """
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super().__init__(input_shape, n_actions)
        state_size = input_shape[-1]

        conv_out_size = self.get_conv_size(input_shape)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 32),
            nn.ReLU(),
            nn.Linear(32, state_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Uses the layers from BaseModel (except output layer) to obtain an encoded state of
        shape -> (batch_size, state_size)."""
        conv_out = self.conv(state).view(state.size(0), -1)  # shape -> (batch_size, flatten_size)
        x = self.fc_layers(conv_out)
        return x  # shape -> (batch_size, state_size)


class EmpowermentModel(nn.Module):
    """A base empowerment model used across its family on networks."""
    def __init__(self, state_size: int, n_actions: int, device: str) -> None:
        super().__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.device = device

    def one_hot_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Helper function that creates a matrix of one hot encoded action probabilities,
        where the action position = 1 and other values are 0."""
        one_hot_actions = torch.zeros(actions.shape[0], self.n_actions, device=self.device)
        indices = torch.stack((torch.arange(actions.shape[0], device=self.device), actions), dim=0).tolist()
        one_hot_actions[indices] = 1.
        return one_hot_actions  # shape -> (batch_size, n_actions)


class SourceNet(EmpowermentModel):
    """A network used for predicting an action to take given a current state.

    :param state_size (int) - size of the input nodes
    :param n_actions (int) - number of possible actions in the environment
    :param device (str) - name of CUDA device
    """
    def __init__(self, state_size: int, n_actions: int, device: str) -> None:
        super().__init__(state_size, n_actions, device)

        self.layers = nn.Sequential(
            nn.Linear(state_size + n_actions, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes action prediction from given encoded state and one-hot encoded actions.

        :param x (torch.Tensor) - a combination of an encoded state and one hot encoded actions,
                                  shape -> (batch_size, state_size + n_actions)
        """
        return self.layers(x)  # shape -> (batch_size, 1)


class ForwardDynamicsNet(EmpowermentModel):
    """
    A forward dynamics network for computing a next state prediction.

    :param state_size (int) - size of the input nodes
    :param n_actions (int) - number of possible actions in the environment
    :param device (str) - name of CUDA device
    """
    def __init__(self, state_size: int, n_actions: int, device: str) -> None:
        super().__init__(state_size, n_actions, device)

        self.layers_top = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.layers_bot = nn.Sequential(
            nn.Linear(64 + n_actions, 64),
            nn.ReLU(),
            nn.Linear(64, state_size)
        )

    def forward(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Computes a next state prediction given an encoded state and a set of actions taken in that state.
        """
        x = self.layers_top(encoded_state)
        one_hot_actions = self.one_hot_actions(actions)
        x = torch.cat((x, one_hot_actions), dim=1)
        return self.layers_bot(x)  # shape -> (batch_size, state_size)


class EmpowerModel(nn.Module):
    """A base class for agent models that are using empowerment."""
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super().__init__()
        self.n_actions = n_actions

        self.layers = nn.Sequential(
            nn.Linear(input_shape[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )


class QNetwork(EmpowerModel):
    """
    A linear Q-network that converts an encoded state into action probabilities.

    :param input_shape (tuple[int]) - image input dimensions (including batch size at first dimension)
    :param n_actions (int) - number of possible actions in the environment
    """
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super().__init__(input_shape, n_actions)

        self.out = nn.Linear(32, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the action probabilities for an encoded state."""
        x = self.layers(x)
        x = self.out(x)
        return x  # shape -> (batch_size, n_actions)


class RainbowNetwork(EmpowerModel):
    """
    A noisy linear Rainbow Q-network that converts an encoded state into action probabilities.

    :param input_shape (tuple[int]) - image input dimensions (including batch size at first dimension)
    :param n_actions (int) - number of possible actions in the environment
    :param n_atoms (int) - number of distributions
    """
    def __init__(self, input_shape: tuple, n_actions: int, n_atoms: int = 51) -> None:
        super().__init__(input_shape, n_actions)
        self.n_atoms = n_atoms

        self.layers = nn.Sequential(
            nn.Linear(input_shape[-1], 80),  # Input non-noisy
            nn.ReLU(),
            NoisyLinear(80, 64),
            nn.ReLU()
        )
        self.advantage = NoisyLinear(64, n_actions * n_atoms)
        self.value = NoisyLinear(64, n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the action probabilities for an encoded state."""
        fc_out = self.layers(x)

        advantage = self.advantage(fc_out)
        value = self.value(fc_out)

        # Reshape to accommodate atoms
        advantage = advantage.reshape(-1, self.n_actions, self.n_atoms)  # shape -> (batch_size, n_actions, n_atoms)
        value = value.reshape(-1, 1, self.n_atoms)  # shape -> (batch_size, 1, n_atoms)

        # Compute Q-values
        q = value + advantage - advantage.mean()  # shape -> (batch_size, n_actions, n_atoms)
        q = F.softmax(q, dim=2)  # Probabilities of actions over atoms
        return q  # shape -> (batch_size, n_actions, n_atoms)

    def sample_noise(self, device: str) -> None:
        """Samples new noise in the NoisyLinear layers."""
        self.layers[2].sample_noise()
        self.advantage.sample_noise()
        self.value.sample_noise()
        self.to(device)


class PPONetwork(EmpowerModel):
    """A linear actor-critic network that converts an encoded state into action probabilities and state-values."""
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super().__init__(input_shape, n_actions)

        self.actor = nn.Linear(32, n_actions)
        self.critic = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """Computes the action probabilities and state-values for an encoded state."""
        x = self.layers(x)
        return F.softmax(self.actor(x), dim=1), self.critic(x)
