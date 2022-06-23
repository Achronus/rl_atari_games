from dataclasses import dataclass

from models._base import BaseModel

import torch
import torch.optim as optim
import torch.nn as nn


@dataclass
class EnvParameters:
    """A data class for environment parameters."""
    env_name: str = ""
    img_size: int = 128
    stack_size: int = 4
    capture_video: bool = False
    record_every: int = 100
    seed: int = 1


@dataclass
class ModelParameters:
    """A data class for model (neural network) parameters."""
    network: BaseModel = BaseModel
    optimizer: optim.Optimizer = optim.Optimizer
    loss_metric: nn.modules.loss = nn.MSELoss()


class AgentParameters:
    """A base agent parameters class, strictly for inheriting from."""
    pass


@dataclass
class CoreDQNParameters(AgentParameters):
    """A data class containing the core DQN parameters."""
    gamma: float = 0.99  # Discount factor
    tau: float = 0.001  # Soft updater for target network
    update_steps: int = 4  # How often to update the network
    max_timesteps: int = 1000  # Max before episode end


@dataclass
class DQNParameters(CoreDQNParameters):
    """A data class for DQN parameters."""
    buffer_size: int = 10  # Size of memory buffer
    batch_size: int = 1  # Buffer mini-batch size
    eps_start: float = 1.0  # Initial epsilon
    eps_end: float = 0.01  # Greedy epsilon threshold
    eps_decay: float = 0.995  # Epsilon decay rate


@dataclass
class RainbowDQNParameters(CoreDQNParameters):
    """A data class for Rainbow DQN parameters."""
    replay_period: int = 100  # Number of transitions before learning begins
    n_steps: int = 3  # Number of steps for multi-step learning
    learn_frequency: int = 4  # Number of timesteps to perform agent learning
    clip_grad: float = 0.5  # Maximum value for gradient clipping
    reward_clip: float = 0.1  # Number for maximum reward bounds

    # Categorical DQN
    n_atoms: int = 51  # Number of atoms
    v_min: int = -10  # Minimum size of the atoms
    v_max: int = 10  # Maximum size of the atoms


@dataclass
class PPOParameters(AgentParameters):
    """A data class for PPO parameters."""
    gamma: float = 0.99  # Discount factor
    update_steps: int = 4  # How often to update the network
    loss_clip: float = 0.5  # Value for surrogate clipping
    rollout_size: int = 100  # Number of samples to train on
    num_agents: int = 4  # Number of agents used during training
    num_mini_batches: int = 4  # Number of mini-batches during training
    entropy_coef: float = 0.01  # Coefficient for regularisation
    value_loss_coef: float = 0.5  # Coefficient for decreasing value loss
    clip_grad: float = 0.5  # Maximum value for gradient clipping


@dataclass
class BufferParameters:
    """A data class containing the parameters for Prioritized Experience Replay Buffers."""
    buffer_size: int = 100  # Size of memory buffer
    batch_size: int = 32  # Buffer mini-batch size
    priority_exponent: float = 0.05  # Prioritized buffer exponent (alpha)
    priority_weight: float = 0.4  # Initial prioritized buffer importance sampling weight (beta)
    input_shape: tuple = (128, 128)  # State image input shape, obtained from the environment


@dataclass
class Experience:
    """A data class to store transitions (experiences)."""
    state: torch.Tensor
    action: int
    reward: float
    done: bool
    next_state: torch.Tensor = torch.zeros((1, 1))
