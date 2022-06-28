from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class IMParameters:
    """A parent data class containing core intrinsic motivation parameters."""
    input_shape: tuple = (128, 128)
    n_actions: int = 6


@dataclass
class CuriosityParameters(IMParameters):
    """A data class to store the curiosity intrinsic motivation's parameters."""
    forward_loss: nn.modules.loss = nn.MSELoss(reduction='none')
    inverse_loss: nn.modules.loss = nn.CrossEntropyLoss(reduction='none')
    comparison_weight: float = 0.2  # weighs inverse model loss against forward model loss (beta)
    importance_weight: float = 0.1  # weights importance of policy gradient loss vs learning reward signal (lambda)
    scaling_factor: float = 1.  # a scaling factor for the reward (eta)


@dataclass
class EmpowermentParameters(IMParameters):
    """A data class to store the empowerment intrinsic motivation's parameters."""
    pass


@dataclass
class SurpriseBasedParameters(IMParameters):
    """A data class to store the surprise-based intrinsic motivation parameters."""
    pass


@dataclass
class IMExperience:
    """A data class that stores transition values required for intrinsic motivation methods."""
    state: torch.Tensor = torch.zeros(1, 128, 128)
    next_state: torch.Tensor = torch.zeros(1, 128, 128)
    actions: torch.Tensor = torch.ones(1, 6)
