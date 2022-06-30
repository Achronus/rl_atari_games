from models._base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class CuriosityModel(BaseModel):
    """A neural network for the curiosity intrinsic motivation method. Based on the Curiosity-driven Exploration paper:
       https://arxiv.org/abs/1705.05363.

    :param input_shape (tuple[int]) - image input dimensions (including batch size at first dimension)
    :param n_actions (int) - number of possible actions in the environment
    :param device (str) - name of CUDA device
    """
    def __init__(self, input_shape: tuple, n_actions: int, device: str) -> None:
        super().__init__(input_shape, n_actions)
        self.device = device

        # Core dense layers
        self.fc_core = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        conv_out_size = self.get_conv_size(input_shape)

        # Action prediction (predict_action)
        self.fc_in = nn.Linear(conv_out_size * 2, 256)

        # State prediction (forward)
        self.state_pred_in = nn.Linear(conv_out_size + n_actions, 256)
        self.state_pred_out = nn.Linear(128, conv_out_size)

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
        # Create matrix of action probabilities -> action position = 1, else 0
        actions_ = torch.zeros(actions.shape[0], self.n_actions, device=self.device)  # shape -> (batch_size, n_actions)
        indices = torch.stack((torch.arange(actions.shape[0], device=self.device), actions), dim=0).tolist()
        actions_[indices] = 1.  # one-hot encoded matrix

        # Combine state and matrix of actions and pass through dense layers
        x = torch.cat((encoded_state, actions_), dim=1)  # shape -> (batch_size, flatten_size + n_actions)
        x = self.state_pred_in(x)
        x = self.fc_core(x)
        return self.state_pred_out(x)  # shape -> (batch_size, flatten_size)


class EmpowermentModel(BaseModel):
    """A neural network for the empowerment intrinsic motivation method. Based on the X paper:

    :param input_shape (tuple[int]) - image input dimensions (including batch size at first dimension)
    :param n_actions (int) - number of possible actions in the environment
    """
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super().__init__(input_shape, n_actions)


class SurpriseBasedModel(BaseModel):
    """A neural network for the surprise-based intrinsic motivation method. Based on the X paper:

    :param input_shape (tuple[int]) - image input dimensions (including batch size at first dimension)
    :param n_actions (int) - number of possible actions in the environment
    """
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super().__init__(input_shape, n_actions)
