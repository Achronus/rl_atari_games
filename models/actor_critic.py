from models.cnn import CNNModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(CNNModel):
    """
    An Actor model used for PPO.

    Parameters:
        input_shape (tuple[int]) - image input dimensions (including batch size)
        n_actions (int) - number of possible actions in the environment
        seed (int) - a number for recreating results, applied to random operations
    """
    def __init__(self, input_shape: tuple[int, ...], n_actions: int, seed: int) -> None:
        super().__init__(input_shape, n_actions, seed)

        conv_out_size = self.get_conv_size(input_shape)

        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the features forward through the network."""
        conv_out = self.conv(x).view(x.size(0), -1)
        return F.softmax(self.actor(conv_out), dim=1)


class Critic(CNNModel):
    """
    A Critic model used for PPO.

    Parameters:
        input_shape (tuple[int]) - image input dimensions (including batch size)
        n_actions (int) - number of possible actions in the environment
        seed (int) - a number for recreating results, applied to random operations
    """
    def __init__(self, input_shape: tuple[int, ...], n_actions: int, seed: int) -> None:
        super().__init__(input_shape, n_actions, seed)

        conv_out_size = self.get_conv_size(input_shape)

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the features forward through the network."""
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.critic(conv_out)
