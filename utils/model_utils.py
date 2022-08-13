import os
import numpy as np

from agents._agent import Agent
from agents.dqn import DQN
from agents.ppo import PPO
from agents.rainbow import RainbowDQN
from utils.dataloader import DataLoader
from core.enums import ValidModels
from utils.helper import to_tensor, normalize


def load_model(filename: str, device: str, model_type: str) -> Agent:
    """
    Load a DQN or PPO model based on its given filename.

    :param filename (str) - filename of the model to load (must be stored in a 'saved_models' folder)
    :param device (str) - the CUDA device to load the model onto (CPU or GPU)
    :param model_type (str) - the type of model to load (DQN or PPO)
    """
    filename = filename if filename[-3:] == '.pt' else f'{filename}.pt'
    assert os.path.exists('saved_models'), "'saved_models' folder does not exist! Have you created it?"
    assert os.path.exists(f'saved_models/{filename}'), f"'{filename}' does not exist in the 'saved_models' folder!"

    model = model_type.lower()
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


def test_model(model: Agent, episodes: int = 100) -> list:
    """Tests a given model and returns a list of episode scores."""
    env = model.env_details.make_env('testing')
    ep_scores = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0
        while not done:
            state = normalize(to_tensor(state)).to(model.device)
            if type(model) == DQN or type(model) == RainbowDQN:
                if type(model) == RainbowDQN:
                    state = state.unsqueeze(0)
                state = model.encode_state(state)
                action = model.act(state)  # Generate an action
            elif type(model) == PPO:
                state = model.encode_state(state.unsqueeze(0))
                action_probs, _ = model.network.forward(state)
                preds = model.act(action_probs)  # Generate an action
                action = preds['action'].item()

            next_state, reward, done, info = env.step(action)  # Take an action
            state = next_state
            score += reward

        ep_scores.append(score)
        if episode % 10 == 0 or episode == 1 or episode == episodes+1:
            print(f'({episode}/{episodes}) Episode score: {int(score)}')
    env.close()
    print('Scores avg:', np.mean(np.asarray(ep_scores)))
    return ep_scores
