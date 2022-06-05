from dataclasses import dataclass

from models._base import BaseModel

import torch.optim as optim
import torch.nn.modules as modules


@dataclass
class ModelParameters:
    """A data class for model (neural network) parameters."""
    network: BaseModel
    optimizer: optim.Optimizer
    loss_metric: modules.loss


@dataclass
class DQNParameters:
    """A data class for DQN parameters."""
    gamma: float  # Discount factor
    tau: float  # Soft updater for target network
    buffer_size: int  # Replay buffer size
    batch_size: int  # Buffer training batch size
    update_steps: int  # How often to update the network
    eps_start: float = 1.0  # Initial epsilon
    eps_end: float = 0.01  # Greedy epsilon threshold
    eps_decay: float = 0.995  # Epsilon decay rate
    max_timesteps: int = 1000  # Max before episode end
