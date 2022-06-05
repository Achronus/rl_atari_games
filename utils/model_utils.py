import os
from typing import Union

from agents.dqn import DQN
from agents.ppo import PPO
from core.parameters import ModelParameters
from models.cnn import CNNModel

import torch


def load_model(filename: str, device: str, model_type: str) -> Union[DQN, PPO]:
    """Load a DQN or PPO model based on its given filename."""
    assert os.path.exists('saved_models'), "'saved_models' folder does not exist! Have you created it?"
    assert os.path.exists(f'saved_models/{filename}.pt'), "'filename' does not exist in the 'saved_models' folder!"

    if model_type.lower() == 'dqn':
        return __load_dqn_model(filename, device)
    elif model_type.lower() == 'ppo':
        return __load_ppo_model(filename, device)
    else:
        raise ValueError(f"Model type '{model_type}' does not exist! Must be either 'DQN' or 'PPO'")


def __load_dqn_model(filename: str, device: str) -> DQN:
    """Load a DQN model's parameters from the given filename. Files must be stored within a saved_models folder."""
    checkpoint = torch.load(f'saved_models/{filename}.pt', map_location=device)
    env_details = checkpoint.get('env_details')
    dqn_params = checkpoint.get('dqn_params')
    logger = checkpoint.get('logger')
    seed = checkpoint.get('seed')

    model_params = ModelParameters(
        network=CNNModel(input_shape=env_details.input_shape, n_actions=env_details.n_actions, seed=seed),
        optimizer=checkpoint.get('optimizer'),
        loss_metric=checkpoint.get('loss_metric')
    )

    dqn = DQN(env_details, model_params, dqn_params, seed)
    dqn.local_network.load_state_dict(checkpoint.get('local_network'), strict=False)
    dqn.target_network.load_state_dict(checkpoint.get('target_network'), strict=False)
    dqn.logger = logger
    print(f"Loaded DQN model: '{filename}'.")
    return dqn


def __load_ppo_model(filename: str, device: str) -> PPO:
    """Load a PPO model's parameters from the given filename. Files must be stored within a saved_models folder."""
    pass
