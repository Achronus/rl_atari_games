import os

from typing import Union

from agents.dqn import DQN
from agents.rainbow import RainbowDQN
from agents.ppo import PPO
from utils.dataloader import DataLoader


def load_model(filename: str, device: str, model_type: str) -> Union[DQN, PPO, RainbowDQN]:
    """
    Load a DQN or PPO model based on its given filename.

    Parameters:
        filename (str) - filename of the model to load (must be stored in a 'saved_models' folder)
        device (str) - the CUDA device to load the model onto (CPU or GPU)
        model_type (str) - the type of model to load (DQN or PPO)
    """
    file = filename if filename[-3:] == '.pt' else f'{filename}.pt'
    assert os.path.exists('saved_models'), "'saved_models' folder does not exist! Have you created it?"
    assert os.path.exists(f'saved_models/{file}'), f"'{file}' does not exist in the 'saved_models' folder!"

    model = model_type.lower()
    valid_models = ['DQN', 'PPO', 'rainbow']
    loader = DataLoader(filename, device)

    # Load desired model
    if model == 'dqn':
        loaded_model = loader.load_dqn_model()
    elif model == 'ppo':
        loaded_model = loader.load_ppo_model()
    elif model == 'rainbow':
        loaded_model = loader.load_rdqn_model()
    else:
        raise ValueError(f"Model type '{model_type}' does not exist! Must be one of: {valid_models}.")

    # Update model logger if data available
    env_name = loaded_model.save_file_env_name()
    logger_data = loader.unpack_logger_data(env_name)
    if logger_data is not None:
        loaded_model.logger = logger_data

    print(f"Loaded model: '{loader.filename}'")
    return loaded_model
