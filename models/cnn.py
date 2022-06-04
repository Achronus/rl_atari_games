import numpy as np

from models._base import BaseModel

import torch
import torch.nn as nn


class CNNModel(BaseModel):
    """
    A CNN model used for all agents.

    Parameters:
        input_shape (tuple[int]) - image input dimensions (including batch size)
        n_actions (int) - number of possible actions in the environment
        seed (int) - a number for recreating results, applied to random operations
    """
    def __init__(self, input_shape: tuple[int, ...], n_actions: int, seed: int) -> None:
        super().__init__(input_shape, n_actions)
        self.seed = torch.manual_seed(seed)

        # Convolutional Layers
        # Comments assume input_shape: (4, 128, 128)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=5, stride=2),  # 128x128x4 -> 62x62x64
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # 62x62x64 -> 60x60x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 60x60x64 -> 30x30x64

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),  # 30x30x64 -> 28x28x32
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),  # 28x28x32 -> 26x26x32
            nn.MaxPool2d(3),  # 26x26x32 -> 8x8x32
        )
        conv_out_size = self.get_conv_size(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def get_conv_size(self, input_shape: tuple[int, ...]) -> int:
        """Returns the convolutional layers output size."""
        out = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(out.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates the features forward through the network."""
        conv_out = self.conv(x).view(x.size(0), -1)  # 8x8x32 -> 2,048
        return self.fc(conv_out)  # Q-values
