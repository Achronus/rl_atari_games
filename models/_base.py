import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    A basic model used for neural network architectures.

    Parameters:
        input_shape (tuple[int]) - image input dimensions (including batch size)
        n_actions (int) - number of possible actions in the environment
    """
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        self.fc = nn.Sequential(
            nn.Linear(input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the supplied features forward through the network."""
        return F.softmax(self.fc(x), dim=1)
