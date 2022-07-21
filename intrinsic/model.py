from intrinsic._im_base import IMBaseModel
from intrinsic.empower_models import Encoder, SourceNet, ForwardDynamicsNet

import torch
import torch.nn as nn
import torch.nn.functional as F


class CuriosityModel(IMBaseModel):
    """A neural network for the curiosity intrinsic motivation method. Based on the Curiosity-driven Exploration paper:
       https://arxiv.org/abs/1705.05363.

    :param input_shape (tuple[int]) - image input dimensions (including batch size at first dimension)
    :param n_actions (int) - number of possible actions in the environment
    :param device (str) - name of CUDA device
    """
    def __init__(self, input_shape: tuple, n_actions: int, device: str) -> None:
        super().__init__(input_shape, n_actions, device)
        # Core dense layers
        self.fc_core = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Action prediction (predict_action)
        self.fc_in = nn.Linear(self.conv_out_size * 2, 256)

        # State prediction (forward)
        self.state_pred_in = nn.Linear(self.conv_out_size + n_actions, 256)
        self.state_pred_out = nn.Linear(128, self.conv_out_size)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encodes a given state, reducing its dimensionality and flattening it. Returns the 1D state."""
        conv_out = self.conv(state).view(state.size(0), -1)  # shape -> (batch_size, flatten_size)
        return conv_out

    def predict_action(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Computes action probability predictions for the action taken from a given state to reach
        the given next state."""
        encoded_state = self.encode(state)
        encoded_next_state = self.encode(next_state)

        # Combine flattened encoded states and pass through dense layers
        x = torch.cat((encoded_state, encoded_next_state), dim=1)
        x = self.fc_in(x)
        x = self.fc_core(x)
        return F.softmax(self.out(x), dim=1)  # shape -> (batch_size, n_actions)

    def forward(self, encoded_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute the prediction error (curiosity signal).

        :param encoded_state (torch.Tensor) - an encoded state, shape -> (batch_size, flatten_size)
        :param actions (torch.Tensor) - a batch of actions to take, shape -> (batch_size)
        """
        # One-hot encode actions
        one_hot_actions = self.one_hot_actions(actions)

        # Combine state and matrix of actions and pass through dense layers
        x = torch.cat((encoded_state, one_hot_actions), dim=1)  # shape -> (batch_size, flatten_size + n_actions)
        x = self.state_pred_in(x)
        x = self.fc_core(x)
        return self.state_pred_out(x)  # shape -> (batch_size, flatten_size)


class EmpowermentModel(IMBaseModel):
    """A set of neural networks for the empowerment intrinsic motivation method. Based on the Empowerment-driven
    Exploration using Mutual Information Estimation paper: https://arxiv.org/abs/1810.05533.

    :param input_shape (tuple[int]) - image input dimensions (including batch size at first dimension)
    :param n_actions (int) - number of possible actions in the environment
    :param device (str) - name of CUDA device
    """
    def __init__(self, input_shape: tuple, n_actions: int, device: str) -> None:
        super().__init__(input_shape, n_actions, device)

        self.encoder = Encoder(input_shape, n_actions).to(device)
        self.source_net = SourceNet(input_shape[-1], n_actions, device).to(device)
        self.forward_net = ForwardDynamicsNet(input_shape[-1], n_actions, device).to(device)

        self.source_target = SourceNet(input_shape[-1], n_actions, device).to(device)
        self.forward_target = ForwardDynamicsNet(input_shape[-1], n_actions, device).to(device)

        # Fix encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
