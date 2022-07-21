from models._base import BaseModel
from models.linear import NoisyLinear

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalNoisyDueling(BaseModel):
    """
    A Noisy Dueling Deep Q-Network with value distribution. Combines Noisy networks, Dueling network, and
    Distributional Value Learning (C51/Categorical DQN). The implementation is based the Noisy Nets for Exploration,
    Dueling Network Architectures, and Distributional Perspective papers, respectively.

    Noisy Nets for Exploration paper: https://arxiv.org/pdf/1706.10295v3.pdf.
    Dueling Network Architectures paper: https://arxiv.org/pdf/1511.06581.pdf.
    Distributional Perspective paper: https://arxiv.org/pdf/1707.06887.pdf.

    :param input_shape (tuple[int]) - image input dimensions (including batch size)
    :param n_actions (int) - number of possible actions in the environment
    :param n_atoms (int) - number of distributions
    """
    def __init__(self, input_shape: tuple, n_actions: int, n_atoms: int = 51) -> None:
        super().__init__(input_shape, n_actions)
        self.n_atoms = n_atoms
        conv_out_size = self.get_conv_size(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),  # Input non-noisy
            nn.ReLU(),
            NoisyLinear(256, 128),
            nn.ReLU()
        )
        self.advantage = NoisyLinear(128, n_actions * n_atoms)
        self.value = NoisyLinear(128, n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the features forward through the network. Returns the action probabilities
        computed from the atoms advantage and value branches."""
        conv_out = self.conv(x).view(x.size(0), -1)
        fc_out = self.fc(conv_out)

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
        self.fc[2].sample_noise()
        self.advantage.sample_noise()
        self.value.sample_noise()
        self.to(device)
