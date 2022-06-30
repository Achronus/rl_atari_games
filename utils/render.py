from IPython import display

import matplotlib.pyplot as plt

from agents._agent import Agent
from agents.dqn import DQN
from agents.ppo import PPO
from agents.rainbow import RainbowDQN
from utils.helper import to_tensor, normalize


def video_render(agent: Agent, episodes: int = 5) -> None:
    """Watch a video representation of an agent in a given environment."""
    env = agent.env_details.make_env('testing', visualize=True)

    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0
        while not done:
            state = normalize(to_tensor(state)).to(agent.device)
            if type(agent) == DQN or type(agent) == RainbowDQN:
                if type(agent) == RainbowDQN:
                    state = state.unsqueeze(0)
                action = agent.act(state)  # Generate an action
            elif type(agent) == PPO:
                action_probs, _ = agent.network.forward(state.unsqueeze(0))
                preds = agent.act(action_probs)  # Generate an action
                action = preds['action'].item()

            next_state, reward, done, info = env.step(action)  # Take an action
            state = next_state
            score += reward

        print(f'({episode}/{episodes}) Score: {int(score)}')
    env.close()


def plot_render(agent: Agent, episodes: int = 5) -> None:
    """EPILEPSY WARNING: without the correct settings, this plot flickers in Jupyter Notebooks.
    It is highly recommended to use 'video_render' instead.

    Watch a plot representation of an agent in a given environment."""
    state = agent.env.reset()
    img = plt.imshow(agent.env.render(mode='rgb_array'))  # only call this once

    for episode in range(1, episodes+1):
        done = False
        score = 0
        while not done:
            img.set_data(agent.env.render(mode='rgb_array'))  # just update the data
            display.display(plt.gcf())
            display.clear_output(wait=True)

            state = normalize(to_tensor(state)).to(agent.device)
            if type(agent) == DQN or type(agent) == RainbowDQN:
                if type(agent) == RainbowDQN:
                    state = state.unsqueeze(0)
                action = agent.act(state)  # Generate an action
            elif type(agent) == PPO:
                action_probs, _ = agent.network.forward(state.unsqueeze(0))
                preds = agent.act(action_probs)  # Generate an action
                action = preds['action'].item()

            next_state, reward, done, info = agent.env.step(action)  # Take an action
            state = next_state
            score += reward

            if done:
                state = agent.env.reset()
        print(f'({episode}/{episodes}) Score: {int(score)}')
    agent.env.close()
