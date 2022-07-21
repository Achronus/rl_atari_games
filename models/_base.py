import numpy as np

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    A base CNN model used for all agents. Uses identical convolutions as seen in the Human-level Control paper:
    https://www.nature.com/articles/nature14236.

    :param input_shape (tuple[int]) - image input dimensions (including batch size)
    :param n_actions (int) - number of possible actions in the environment
    """
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.n_actions = n_actions

        # Convolutional Layers
        # Comments assume input_shape: (4, 128, 128)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),  # 128x128x4 -> 31x31x32
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # 31x31x32 -> 14x14x64
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # 14x14x64 -> 12x12x64
            nn.ReLU()
        )
        conv_out_size = self.get_conv_size(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.out = nn.Linear(128, n_actions)

    def get_conv_size(self, input_shape: tuple) -> int:
        """Returns the convolutional layers output size."""
        if len(input_shape) < 3:
            out = self.conv(torch.zeros(1, *input_shape))
        else:
            out = self.conv(torch.zeros(*input_shape))
        return int(np.prod(out.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the features forward through the network."""
        conv_out = self.conv(x).view(x.size(0), -1)  # 12x12x64 -> 9,216
        fc_out = self.fc(conv_out)
        return self.out(fc_out)
