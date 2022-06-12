import os
import random
from typing import Union
from dotenv import load_dotenv
import numpy as np

from agents.dqn import DQN
from agents.ppo import PPO
from core.env_details import EnvDetails
from core.exceptions import MissingVariableError
from core.parameters import (
    EnvParameters,
    DQNParameters,
    PPOParameters,
    ModelParameters
)
from models.actor_critic import ActorCritic
from models.cnn import CNNModel

import torch
import torch.optim as optim
import torch.nn as nn


def create_model(model_type: str) -> Union[DQN, PPO]:
    """Initializes predefined parameters from a .env file and creates a model of the specified type.
    Returns the model as a class instance."""
    assert os.path.exists('.env'), f"'.env' file does not exist! Have you created it in '{os.getcwd()}'?"

    # Check initial parameters are stored in .env
    set_model = SetModels()
    seed = set_model.seed

    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create selected model
    if model_type.lower() == 'dqn':
        return set_model.create_dqn()
    elif model_type.lower() == 'ppo':
        return set_model.create_ppo()
    else:
        raise ValueError(f"Model type '{model_type}' does not exist! Must be either 'DQN' or 'PPO'")


class SetModels:
    """A class that sets the parameters from a predefined .env file and allows creation of model instances."""
    generic_params = ['SEED', 'GAMMA', 'LEARNING_RATE', 'EPSILON', 'UPDATE_STEPS']
    env_params = ['ENV_1', 'IMG_SIZE', 'STACK_SIZE', 'CAPTURE_VIDEO', 'SAVE_EVERY']
    dqn_params = ['TAU', 'BUFFER_SIZE', 'BATCH_SIZE', 'EPS_START', 'EPS_END', 'EPS_DECAY',
                  'MAX_TIMESTEPS']
    ppo_params = ['CLIP_GRAD', 'ROLLOUT_SIZE', 'NUM_AGENTS', 'NUM_MINI_BATCHES', 'ENTROPY_COEF',
                  'VALUE_LOSS_COEF', 'MAX_GRAD_NORM']

    def __init__(self) -> None:
        load_dotenv()  # Create access to .env file

        # Check generic and env params exist
        self.check_params(self.generic_params + self.env_params)

        self.seed = int(os.getenv('SEED'))
        self.lr = float(os.getenv('LEARNING_RATE'))
        self.eps = float(os.getenv('EPSILON'))
        self.capture_video = True if os.getenv('CAPTURE_VIDEO') == 'True' else False
        self.save_every = int(os.getenv('SAVE_EVERY'))

        self.env_details = self.__create_env_details()

    @staticmethod
    def check_params(param_list: list[str]) -> None:
        """Checks if the given parameter type are set in the '.env' file."""
        # Handle missing parameters
        false_bools = [idx for idx, item in enumerate(param_list) if item not in os.environ]
        if len(false_bools) >= 1:
            missing_params = [param_list[i] for i in false_bools]
            raise MissingVariableError(f"Cannot find variables {missing_params} in .env file! Have you added them?")

    def __create_env_details(self) -> EnvDetails:
        """Creates environment details class."""
        env_params = EnvParameters(
            env_name=os.getenv('ENV_1'),
            img_size=int(os.getenv('IMG_SIZE')),
            stack_size=int(os.getenv('STACK_SIZE')),
            capture_video=self.capture_video,
            record_every=self.save_every,
            seed=self.seed
        )
        return EnvDetails(env_params)

    def create_dqn(self) -> DQN:
        """Creates DQN model from .env predefined parameters."""
        self.check_params(self.dqn_params)

        network = CNNModel(input_shape=self.env_details.input_shape,
                           n_actions=self.env_details.n_actions)

        model_params = ModelParameters(
            network=network,
            optimizer=optim.Adam(network.parameters(), lr=self.lr, eps=self.eps),
            loss_metric=nn.MSELoss()
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

    def create_ppo(self) -> PPO:
        """Creates a PPO model from .env predefined parameters."""
        self.check_params(self.ppo_params)

        ac = ActorCritic(input_shape=self.env_details.input_shape,
                         n_actions=self.env_details.n_actions)

        model_params = ModelParameters(
            network=ac,
            optimizer=optim.Adam(ac.parameters(), lr=self.lr, eps=self.eps),
            loss_metric=nn.MSELoss()
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
