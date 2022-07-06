from enum import Enum

from intrinsic.module import Curiosity, Empowerment, SurpriseBased
from intrinsic.model import CuriosityModel, EmpowermentModel, SurpriseBasedModel

import torch.optim as optim


class CoreCheckpointParams(Enum):
    """An enum that stores the core checkpoint (save) parameter names.
    Used in the DataLoader class."""
    ENV_DETAILS = 'env_details'
    PARAMS = 'params'
    SEED = 'seed'
    MODEL_PARAMS = 'model_params'
    NETWORK_TYPE = 'network_type'


class OptionalParams(Enum):
    """An enum that stores optional parameters that are ignored from required checks.
    Used in the CheckParamsValid class."""
    forward_loss = 0
    inverse_loss = 1
    input_shape = 2
    record_every = 3
    state_loss = 4


class ValidModels(Enum):
    """An enum that stores the valid model names. Used in the create_model and load_model functions."""
    DQN = 'dqn'
    PPO = 'ppo'
    RAINBOW = 'rainbow'


class ValidIMMethods(Enum):
    """An enum that stores the valid intrinsic motivation method names. Used in the
    create_model function."""
    CURIOSITY = 'curiosity'
    EMPOWERMENT = 'empowerment'
    SURPRISE_BASED = 'surprise_based'


class IMType(Enum):
    """An enum that stores the types of intrinsic motivation methods with corresponding
    components. Used in the IMController class."""
    curiosity = {
        'module': Curiosity,
        'model': CuriosityModel
    }
    empowerment = {
        'module': Empowerment,
        'model': EmpowermentModel,
        'source_optim': optim.Adam,
        'forward_optim': optim.Adam
    }
    surprise_based = {
        'module': SurpriseBased,
        'model': SurpriseBasedModel
    }
