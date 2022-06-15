from dataclasses import dataclass

from models._base import BaseModel

import torch
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
class CoreDQNParameters(AgentParameters):
    """A data class containing the core DQN parameters."""
    gamma: float  # Discount factor
    tau: float  # Soft updater for target network
    buffer_size: int  # Size of memory buffer
    batch_size: int  # Buffer mini-batch size
    update_steps: int  # How often to update the network
    max_timesteps: int = 1000  # Max before episode end


@dataclass
class DQNParameters(CoreDQNParameters):
    """A data class for DQN parameters."""
    eps_start: float = 1.0  # Initial epsilon
    eps_end: float = 0.01  # Greedy epsilon threshold
    eps_decay: float = 0.995  # Epsilon decay rate


@dataclass
class RainbowDQNParameters(CoreDQNParameters):
    """A data class for Rainbow DQN parameters."""
    # Categorical DQN
    n_atoms: int = 51  # Number of atoms
    v_min: int = -10  # Minimum size of the atoms
    v_max: int = 10  # Maximum size of the atoms


@dataclass
class PPOParameters(AgentParameters):
    """A data class for PPO parameters."""
    gamma: float  # Discount factor
    update_steps: int  # How often to update the network
    loss_clip: float  # Value for surrogate clipping
    rollout_size: int  # Number of samples to train on
    num_agents: int = 4  # Number of agents used during training
    num_mini_batches: int = 4  # Number of mini-batches during training
    entropy_coef: float = 0.01  # Coefficient for regularisation
    value_loss_coef: float = 0.5  # Coefficient for decreasing value loss
    clip_grad: float = 0.5  # Maximum value for gradient clipping


@dataclass
class BufferParameters:
    """A data class containing the parameters for Prioritized Experience Replay Buffers."""
    buffer_size: int  # Size of memory buffer
    batch_size: int  # Buffer mini-batch size
    priority_exponent: float  # Prioritized buffer exponent (alpha)
    priority_weight: float  # Initial prioritized buffer importance sampling weight (beta)
    replay_period: int  # Number of transitions before learning begins
    n_steps: int  # Number of steps for multi-step learning


@dataclass
class Experience:
    """A data class to store transitions (experiences)."""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
