import os
from typing import Union

from agents.dqn import DQN, RainbowDQN
from agents.ppo import PPO
from core.parameters import ModelParameters
from models.cnn import CNNModel

import torch


def load_model(filename: str, device: str, model_type: str) -> Union[DQN, PPO, RainbowDQN]:
    """
    Load a DQN or PPO model based on its given filename.

    Parameters:
        filename (str) - filename of the model to load (must be stored in a 'saved_models' folder)
        device (str) - the CUDA device to load the model onto (CPU or GPU)
        model_type (str) - the type of model to load (DQN or PPO)
    """
    assert os.path.exists('saved_models'), "'saved_models' folder does not exist! Have you created it?"
    assert os.path.exists(f'saved_models/{filename}.pt'), "'filename' does not exist in the 'saved_models' folder!"

    model = model_type.lower()
    valid_models = ['DQN', 'PPO', 'rainbow']
    if model == 'dqn':
        return __load_dqn_model(filename, device)
    elif model == 'ppo':
        return __load_ppo_model(filename, device)
    elif model == 'rainbow':
        return __load_rdqn_model(filename, device)
    else:
        raise ValueError(f"Model type '{model_type}' does not exist! Must be one of: {valid_models}.")


def __load_dqn_model(filename: str, device: str) -> DQN:
    """Load a DQN model's parameters from the given filename. Files must be stored within a saved_models folder."""
    cp_data = __get_checkpoint_data(filename, device)

    dqn = DQN(cp_data['env_details'], cp_data['model_params'], cp_data['params'], cp_data['seed'])
    dqn.local_network.load_state_dict(cp_data['other'].get('local_network'), strict=False)
    dqn.target_network.load_state_dict(cp_data['other'].get('target_network'), strict=False)
    dqn.logger = cp_data['logger']
    print(f"Loaded DQN model: '{filename}'.")
    return dqn


def __load_rdqn_model(filename: str, device: str) -> RainbowDQN:
    """Load a DQN model's parameters from the given filename. Files must be stored within a saved_models folder."""
    cp_data = __get_checkpoint_data(filename, device)

    rdqn = RainbowDQN(cp_data['env_details'], cp_data['model_params'], cp_data['params'],
                      cp_data['other'].get('buffer_params'), cp_data['seed'])
    rdqn.local_network.load_state_dict(cp_data['other'].get('local_network'), strict=False)
    rdqn.target_network.load_state_dict(cp_data['other'].get('target_network'), strict=False)
    rdqn.logger = cp_data['logger']
    print(f"Loaded Rainbow DQN model: '{filename}'.")
    return rdqn


def __load_ppo_model(filename: str, device: str) -> PPO:
    """Load a PPO model's parameters from the given filename. Files must be stored within a saved_models folder."""
    cp_data = __get_checkpoint_data(filename, device)

    ppo = PPO(cp_data['env_details'], cp_data['model_params'], cp_data['params'], cp_data['seed'])
    ppo.network.load_state_dict(cp_data['other'].get('network'), strict=False)
    ppo.logger = cp_data['logger']
    print(f"Loaded PPO model: '{filename}'.")
    return ppo


def __get_checkpoint_data(filename: str, device: str) -> dict:
    """Gets the checkpoint data, creates the respective objects and return the info as a dictionary."""
    checkpoint = torch.load(f'saved_models/{filename}.pt', map_location=device)
    env_details = checkpoint.get('env_details')
    core_keys = ['env_details', 'params', 'logger', 'seed', 'model_params']

    return {
        'env_details': env_details,
        'params': checkpoint.get('params'),
        'logger': checkpoint.get('logger'),
        'seed': checkpoint.get('seed'),
        'model_params': ModelParameters(
            network=CNNModel(input_shape=env_details.input_shape, n_actions=env_details.n_actions),
            optimizer=checkpoint.get('optimizer'),
            loss_metric=checkpoint.get('loss_metric')
        ),
        'other': {key: val for key, val in checkpoint.items() if key not in core_keys}
    }
