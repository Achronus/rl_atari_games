from dataclasses import dataclass

from models._base import BaseModel

import torch.optim as optim
import torch.nn.modules as modules


@dataclass
class EnvParameters:
    """A data class for environment parameters."""
    env_name: str
    img_size: int
    stack_size: int
    capture_video: bool = False
    record_every: int = 100
    seed: int = 1


@dataclass
class ModelParameters:
    """A data class for model (neural network) parameters."""
    network: BaseModel
    optimizer: optim.Optimizer
    loss_metric: modules.loss


class AgentParameters:
    """A base agent parameters class, strictly for inheriting from."""
    pass


@dataclass
class DQNParameters(AgentParameters):
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


@dataclass
class PPOParameters(AgentParameters):
    """A data class for PPO parameters."""
    gamma: float  # Discount factor
    update_steps: int  # How often to update the network
    clip_grad: float  # Gradient clipping
    rollout_size: int  # Number of samples to train on
    num_agents: int = 4  # Number of agents used during training
    num_mini_batches: int = 4  # Number of mini-batches during training
    entropy_coef: float = 0.01  # Coefficient for regularisation
    value_loss_coef: float = 0.5  # Coefficient for decreasing value loss
    max_grad_norm: float = 0.5  # Maximum value for gradient clipping
