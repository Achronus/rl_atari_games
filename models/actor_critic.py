from models._base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(BaseModel):
    """
    An Actor-Critic model used for PPO.

    :param input_shape (tuple[int]) - image input dimensions (including batch size)
    :param n_actions (int) - number of possible actions in the environment
    """
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super().__init__(input_shape, n_actions)

        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """Propagates the features forward through the network.
        Returns the action probabilities and state-values from the actor and critic branches."""
        conv_out = self.conv(x).view(x.size(0), -1)
        fc_out = self.fc(conv_out)
        return F.softmax(self.actor(fc_out), dim=1), self.critic(fc_out)
