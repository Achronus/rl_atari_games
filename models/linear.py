import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    A Noisy Linear layer. Implements noise as illustrated in the Noisy Networks for Exploration paper:
    https://arxiv.org/pdf/1706.10295v3.pdf.

    :param in_features (int) - number of input features
    :param out_features (int) - number of output features
    :param std (float) - the standard deviation for initialising the layer (default = 0.5)
    """
    def __init__(self, in_features: int, out_features: int, std: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std = std  # sigma

        weight_tensor = torch.zeros(out_features, in_features)
        self.weight_mu = nn.Parameter(weight_tensor)
        self.weight_sigma = nn.Parameter(weight_tensor)

        bias_tensor = torch.zeros(out_features)
        self.bias_mu = nn.Parameter(bias_tensor)
        self.bias_sigma = nn.Parameter(bias_tensor)

        self.register_buffer('weight_epsilon', weight_tensor)
        self.register_buffer('bias_epsilon', bias_tensor)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> F.linear:
        """Updates the default weights and bias during training with noisy ones.
        Returns an updated linear layer."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def reset_parameters(self) -> None:
        """Reset trainable network parameters, factorized by Gaussian noise."""
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))  # 1 / sqrt(in_features)

        # Set weights
        weight_fill = self.std / math.sqrt(self.weight_sigma.size(1))  # std / sqrt(in_features)
        self.weight_mu.detach().uniform_(-mu_range, mu_range)
        self.weight_sigma.detach().fill_(weight_fill)

        # Set biases
        bias_fill = self.std / math.sqrt(self.bias_sigma.size(0))  # std / sqrt(out_features)
        self.bias_mu.detach().uniform_(-mu_range, mu_range)
        self.weight_sigma.detach().fill_(bias_fill)

    def sample_noise(self) -> None:
        """Creates new noise and updates the epsilon weights and biases."""
        self.weight_epsilon = torch.randn(self.out_features, self.in_features)
        self.bias_epsilon = torch.randn(self.out_features)
