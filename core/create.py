import os
import random
from typing import Union
from dotenv import load_dotenv
import numpy as np

from agents.dqn import DQN, RainbowDQN
from agents.ppo import PPO
from core.env_details import EnvDetails
from core.exceptions import MissingVariableError
from core.parameters import (
    BufferParameters,
    DQNParameters,
    EnvParameters,
    PPOParameters,
    ModelParameters,
    RainbowDQNParameters
)
from models.actor_critic import ActorCritic
from models.cnn import CNNModel
from models.dueling import CategoricalNoisyDueling

import torch
import torch.optim as optim
import torch.nn as nn


def create_model(model_type: str) -> Union[DQN, RainbowDQN, PPO]:
    """Initializes predefined parameters from a .env file and creates a model of the specified type.
    Returns the model as a class instance."""
    valid_names = ['dqn', 'ppo', 'rainbow']
    assert os.path.exists('.env'), f"'.env' file does not exist! Have you created it in '{os.getcwd()}'?"
    if model_type.lower() not in valid_names:
        raise ValueError(f"Model type '{model_type}' does not exist! Must be one of: {valid_names}.")

    load_dotenv()  # Create access to .env file

    # Check initial parameters are stored in .env
    name = model_type.lower()
    CheckParamsValid(name)

    # Create selected model
    set_model = SetModels()
    return set_model.create(name)


class CheckParamsValid:
    """A class that checks if the required parameters are in the .env file."""
    core_params = ['SEED', 'GAMMA', 'LEARNING_RATE', 'EPSILON', 'UPDATE_STEPS']

    def __init__(self, model_type: str) -> None:
        self.env_params = self.__get_env_keys()
        self.general_params = self.core_params + self.env_params
        self.dqn_params = self.get_attribute_names(DQNParameters)
        self.ppo_params = self.get_attribute_names(PPOParameters)
        self.rainbow_params = self.get_attribute_names(RainbowDQNParameters)
        self.buffer_params = self.__get_buffer_keys()

        # Set desired parameters
        params = self.general_params
        if model_type == 'dqn':
            params += self.dqn_params
        elif model_type == 'rainbow':
            params += self.rainbow_params + self.buffer_params
        elif model_type == 'ppo':
            params += self.ppo_params

        # Check parameters exist
        self.check_params(params)

    def __get_env_keys(self) -> list[str]:
        """Gets the environment parameters attribute names as a list and updates certain keys.
        Returns the updated list."""
        keys = self.get_attribute_names(EnvParameters)

        updated_keys = [key.replace(key, 'ENV_1') if key == 'ENV_NAME' else key for key in keys]  # ENV_NAME -> ENV_1
        updated_keys = [key for key in updated_keys if key not in ['RECORD_EVERY', 'SEED']]  # Remove keys
        return updated_keys

    def __get_buffer_keys(self) -> list[str]:
        """Gets the buffer parameters attribute names as a list and updates certain keys.
                Returns the updated list."""
        keys = self.get_attribute_names(BufferParameters)

        updated_keys = [key for key in keys if key not in ['INPUT_SHAPE']]  # Remove keys
        return updated_keys

    @staticmethod
    def check_params(param_list: list[str]) -> None:
        """Checks if the given parameter type are set in the '.env' file."""
        # Handle missing parameters
        false_bools = [idx for idx, item in enumerate(param_list) if item not in os.environ]
        if len(false_bools) >= 1:
            missing_params = [param_list[i] for i in false_bools]
            raise MissingVariableError(f"Cannot find variables {missing_params} in .env file! Have you added them?")

    @staticmethod
    def get_attribute_names(cls) -> list[str]:
        """Gets a list of attribute names for a given class."""
        return [item.upper() for item in vars(cls)['__match_args__']]


class SetModels:
    """A class that sets the parameters from a predefined .env file and creates model instances."""
    def __init__(self) -> None:
        self.seed = int(os.getenv('SEED'))
        self.lr = float(os.getenv('LEARNING_RATE'))
        self.eps = float(os.getenv('EPSILON'))
        self.capture_video = True if os.getenv('CAPTURE_VIDEO') == 'True' else False
        self.__check_record_every()

        # Seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env_details = self.__create_env_details()

    def create(self, model_name: str) -> Union[DQN, RainbowDQN, PPO]:
        """Create a model based on the given name."""
        if model_name == 'dqn':
            return self.__create_dqn()
        elif model_name == 'rainbow':
            return self.__create_rainbow_dqn()
        elif model_name == 'ppo':
            return self.__create_ppo()

    def __check_record_every(self) -> None:
        """Checks if record every is set. Requires capture_video to be true, otherwise default is set."""
        # Handle dependent variables
        if self.capture_video:
            if 'RECORD_EVERY' in os.environ:
                self.record_every = int(os.getenv('RECORD_EVERY'))
            else:
                raise MissingVariableError("Cannot find 'RECORD_EVERY' in .env file! Have you added it?")
        else:
            self.record_every = 1000  # Set default

    def __create_env_details(self) -> EnvDetails:
        """Creates environment details class."""
        env_params = EnvParameters(
            env_name=os.getenv('ENV_1'),
            img_size=int(os.getenv('IMG_SIZE')),
            stack_size=int(os.getenv('STACK_SIZE')),
            capture_video=self.capture_video,
            record_every=self.record_every,
            seed=self.seed
        )
        return EnvDetails(env_params)

    def __create_dqn(self) -> DQN:
        """Creates DQN model from .env predefined parameters."""
        network = CNNModel(input_shape=self.env_details.input_shape,
                           n_actions=self.env_details.n_actions)

        model_params = ModelParameters(
            network=network,
            optimizer=optim.Adam(network.parameters(), lr=self.lr, eps=self.eps)
        )

        params = DQNParameters(
            gamma=float(os.getenv('GAMMA')),
            tau=float(os.getenv('TAU')),
            buffer_size=int(float(os.getenv('BUFFER_SIZE'))),
            batch_size=int(os.getenv('BATCH_SIZE')),
            update_steps=int(os.getenv('UPDATE_STEPS')),
            eps_start=float(os.getenv('EPS_START')),
            eps_end=float(os.getenv('EPS_END')),
            eps_decay=float(os.getenv('EPS_DECAY')),
            max_timesteps=int(os.getenv('MAX_TIMESTEPS'))
        )
        return DQN(self.env_details, model_params, params, self.seed)

    def __create_rainbow_dqn(self) -> RainbowDQN:
        """Creates a Rainbow DQN model from .env predefined parameters."""
        params = RainbowDQNParameters(
            gamma=float(os.getenv('GAMMA')),
            tau=float(os.getenv('TAU')),
            buffer_size=int(float(os.getenv('BUFFER_SIZE'))),
            batch_size=int(os.getenv('BATCH_SIZE')),
            update_steps=int(os.getenv('UPDATE_STEPS')),
            max_timesteps=int(os.getenv('MAX_TIMESTEPS')),
            n_atoms=int(os.getenv('N_ATOMS')),
            v_min=int(os.getenv('V_MIN')),
            v_max=int(os.getenv('V_MAX')),
            replay_period=int(os.getenv('REPLAY_PERIOD')),
            n_steps=int(os.getenv('N_STEPS')),
            learn_frequency=int(os.getenv('LEARN_FREQUENCY')),
            clip_grad=float(os.getenv('CLIP_GRAD')),
            reward_clip=float(os.getenv('REWARD_CLIP'))
        )
        buffer_params = BufferParameters(
            buffer_size=int(float(os.getenv('BUFFER_SIZE'))),
            batch_size=int(os.getenv('BATCH_SIZE')),
            priority_exponent=float(os.getenv('PRIORITY_EXPONENT')),
            priority_weight=float(os.getenv('PRIORITY_WEIGHT')),
            n_steps=int(os.getenv('N_STEPS')),
            input_shape=self.env_details.input_shape
        )
        network = CategoricalNoisyDueling(input_shape=self.env_details.input_shape,
                                          n_actions=self.env_details.n_actions,
                                          n_atoms=params.n_atoms)
        model_params = ModelParameters(
            network=network,
            optimizer=optim.Adam(network.parameters(), lr=self.lr, eps=self.eps)
        )

        return RainbowDQN(self.env_details, model_params, params, buffer_params, self.seed)

    def __create_ppo(self) -> PPO:
        """Creates a PPO model from .env predefined parameters."""
        ac = ActorCritic(input_shape=self.env_details.input_shape,
                         n_actions=self.env_details.n_actions)

        model_params = ModelParameters(
            network=ac,
            optimizer=optim.Adam(ac.parameters(), lr=self.lr, eps=self.eps)
        )

        params = PPOParameters(
            gamma=float(os.getenv('GAMMA')),
            update_steps=int(os.getenv('UPDATE_STEPS')),
            loss_clip=float(os.getenv('LOSS_CLIP')),
            rollout_size=int(os.getenv('ROLLOUT_SIZE')),
            num_agents=int(os.getenv('NUM_AGENTS')),
            num_mini_batches=int(os.getenv('NUM_MINI_BATCHES')),
            entropy_coef=float(os.getenv('ENTROPY_COEF')),
            value_loss_coef=float(os.getenv('VALUE_LOSS_COEF')),
            clip_grad=float(os.getenv('CLIP_GRAD'))
        )
        return PPO(self.env_details, model_params, params, self.seed)


def set_save_every(predefined: int = 1000) -> int:
    """
    Returns an integer value dynamically set based on the 'RECORD_EVERY' environment variable,
    if 'CAPTURE_VIDEO' is True and 'RECORD_EVERY' exists. Otherwise, sets it to the predefined value."""
    load_dotenv()  # Create access to .env file

    capture_video = True if os.getenv('CAPTURE_VIDEO') == 'True' else False
    capture_valid = capture_video and 'RECORD_EVERY' in os.environ
    return int(os.getenv('RECORD_EVERY')) if capture_valid else predefined
