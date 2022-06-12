from models.cnn import CNNModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(CNNModel):
    """
    An Actor-Critic model used for PPO.

    Parameters:
        input_shape (tuple[int]) - image input dimensions (including batch size)
        n_actions (int) - number of possible actions in the environment
    """
    def __init__(self, input_shape: tuple[int, ...], n_actions: int) -> None:
        super().__init__(input_shape, n_actions)

        conv_out_size = self.get_conv_size(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagates the features forward through the network.
        Returns the action probabilities and state-values from the actor and critic branches."""
        conv_out = self.conv(x).view(x.size(0), -1)
        fc_out = self.fc(conv_out)
        return F.softmax(self.actor(fc_out), dim=1), self.critic(fc_out)