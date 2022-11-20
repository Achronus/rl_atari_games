import matplotlib.pyplot as plt
import numpy as np

from agents._agent import Agent
from agents.dqn import DQN
from agents.ppo import PPO
from agents.rainbow import RainbowDQN
from utils.plotter import Plotter
from utils.helper import to_tensor, normalize


def plot_data(data: tuple[list, list], plot_type: str, format_dict: dict, figsize: tuple = (8, 6),
              graph_params: dict = None, filename: str = None) -> None:
    """
    Visualises the x and y data onto the chosen plot type.

    :param data (tuple[list, list]) - lists of data to display on the x-axis and y-axis (x, y)
    :param plot_type (string) - type of plot to create.
                                Available plots: ['line', 'scatter', 'bar', 'histogram', 'boxplot']
    :param format_dict (dict) - a dictionary containing plot formatting information.
                                Required keys: ['title', 'xlabel', 'ylabel']
                                Optional keys: ['disable_y_ticks', 'disable_x_ticks']
    :param figsize (tuple) - (optional) size of the plotted figure.
    :param graph_params (dict) - (optional) additional parameters unique to the selected graph. Refer to matplotlib
                                 documentation for more details.
    :param filename (string) - (optional) a filepath and name for saving the plot. E.g., '/plots/ppo-cur_SpaInv.png'.
    """
    valid_format_keys = ['title', 'xlabel', 'ylabel', 'disable_y_ticks', 'disable_x_ticks']
    if not isinstance(data, tuple):
        raise ValueError("'data' must be a tuple of '(x,)' or ('x, y)' data!")

    if len(data) > 2 or len(data) == 0:
        raise ValueError("Plottable data  must be: '(x,)' or '(x, y)'!")
    elif len(data) == 1 and plot_type in ['line', 'scatter', 'bar']:
        raise ValueError("Missing plottable 'y' data! Must be: 'data=(x, y)'!")
    elif len(data) == 2 and plot_type in ['histogram', 'boxplot']:
        raise ValueError("Too many plottable data points! Must be 'data=(x,)'!")

    for key in format_dict.keys():
        if key not in valid_format_keys:
            raise KeyError(f"Invalid key '{key}'! Required keys: ['title', 'xlabel', 'ylabel'] "
                           f"Optional keys: ['disable_y_ticks', 'disable_x_ticks']")

    fig, ax = plt.subplots(figsize=figsize)
    plotter = Plotter(ax, data, format_dict, graph_params)
    getattr(plotter, plot_type)()  # Create plot

    plt.xticks(rotation=90)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()


def test_model(model: Agent, episodes: int = 100) -> list:
    """Tests a given model and returns a list of episode scores."""
    env = model.env_details.make_env('testing')
    ep_scores = []

    for episode in range(1, episodes + 1):
        state, info = env.reset()
        done, truncated = False, False
        score = 0
        while not done or truncated:
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

            next_state, reward, done, truncated, info = env.step(action)  # Take an action
            state = next_state
            score += reward

        ep_scores.append(score)
        if episode % 10 == 0 or episode == 1 or episode == episodes+1:
            print(f'({episode}/{episodes}) Episode score: {int(score)}')
    env.close()
    print('Scores avg:', np.mean(np.asarray(ep_scores)))
    return ep_scores
