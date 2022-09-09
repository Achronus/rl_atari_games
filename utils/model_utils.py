import os
import re

from agents._agent import Agent
from utils.dataloader import DataLoader
from core.enums import ValidModels
from utils.helper import number_to_num_letter


def load_model(filename: str, device: str) -> Agent:
    """
    Load a DQN or PPO model based on its given filename.

    :param filename (str) - filename of the model to load (must be stored in a 'saved_models' folder)
    :param device (str) - the CUDA device to load the model onto (CPU or GPU)
    """
    filename = filename if filename[-3:] == '.pt' else f'{filename}.pt'
    assert os.path.exists('saved_models'), "'saved_models' folder does not exist! Have you created it?"
    assert os.path.exists(f'saved_models/{filename}'), f"'{filename}' does not exist in the 'saved_models' folder!"

    model_type = filename.split('/')[-1].split('_')[0].lower() if '/' in filename else filename.split('_')[0].lower()
    model = model_type.split('-')[0] if '-' in model_type else model_type
    valid_models = [item.value for item in ValidModels]
    loader = DataLoader(filename, model_type, device)

    # Load desired model
    if model == ValidModels.DQN.value:
        loaded_model = loader.load_dqn_model()
    elif model == ValidModels.PPO.value:
        loaded_model = loader.load_ppo_model()
    elif model == ValidModels.RAINBOW.value:
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


def improve_model(load_model_params: dict, train_params: dict, ep_total: int) -> Agent:
    """
    Loads a trained model and trains it further from an episode start point.

    :param load_model_params (dict) - dictionary containing the load_model() parameters (filename, device)
    :param train_params (dict) - dictionary containing the model.train() parameters (print_every, save_count)
    :param ep_total (int) - number of episodes to reach. E.g., 100,000
    """
    # Get episode start as number
    ep_start_text = load_model_params['filename'].split('_')[-1][2:].lower()

    # Handle start number and letter
    if ep_start_text.isnumeric():
        ep_start_num, ep_start_letter = int(ep_start_text), ''
    else:
        match = re.match(r"([0-9]+)([a-z]+)", ep_start_text, re.I)
        ep_start_num, ep_start_letter = match.groups()

    ep_start = int(ep_start_num) * int(1e3) if ep_start_letter == 'k' else int(ep_start_num)
    eps_remaining = ep_total - ep_start
    model = load_model(**load_model_params)

    ep_s_idx, ep_s_let = number_to_num_letter(ep_start)
    ep_re_idx, ep_re_let = number_to_num_letter(eps_remaining)
    ep_tot_idx, ep_tot_let = number_to_num_letter(ep_total)
    print(f'Starting training from {ep_s_idx}{ep_s_let} episodes for a further {ep_re_idx}{ep_re_let} '
          f'({ep_tot_idx}{ep_tot_let} total).')

    model.train(num_episodes=eps_remaining, custom_ep_start=ep_start, **train_params)
    return model
