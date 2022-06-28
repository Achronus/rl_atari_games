from enum import Enum

from intrinsic.module import Curiosity
from intrinsic.model import CuriosityModel


class CoreCheckpointParams(Enum):
    """An enum that stores the core checkpoint (save) parameter names.
    Used in the DataLoader class."""
    env_details = 0
    params = 1
    logger = 2
    seed = 3
    model_params = 4


class OptionalParams(Enum):
    """An enum that stores optional parameters that are ignored from required checks.
    Used in the CheckParamsValid class."""
    forward_loss = 0
    inverse_loss = 1
    input_shape = 2
    record_every = 3


class ValidModels(Enum):
    """An enum that stores the valid model names. Used in the create_model function."""
    dqn = 0
    ppo = 1
    rainbow = 2


class ValidIMMethods(Enum):
    """An enum that stores the valid intrinsic motivation method names. Used in the
    create_model function."""
    curiosity = 0
    empowerment = 1
    surprise_based = 2


class IMType(Enum):
    """An enum that stores the types of intrinsic motivation methods with corresponding
    components. Used in the IMController class."""
    curiosity = {
        'module': Curiosity,
        'model': CuriosityModel
    }
    empowerment = {
        'loss': 0
    }
    surprise_based = {
        'loss': 0
    }
