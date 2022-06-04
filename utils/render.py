from typing import Union
from IPython import display

import gym
import matplotlib.pyplot as plt

from agents.dqn import DQN
from agents.ppo import PPO
from utils.helper import normalize, to_tensor


def video_render(env: gym.Env, agent: Union[DQN, PPO], steps: int = 3000) -> None:
    """Watch a video representation of an agent in a given environment."""
    state = env.reset()

    for _ in range(steps):
        state = normalize(to_tensor(state)).to(agent.device)
        if type(agent) == DQN:
            action = agent.act(state, 0.01)  # Generate an action
        else:  # PPO
            action = agent.act(state)  # Generate an action
        next_state, reward, done, info = agent.env.step(action)  # Take an action
        env.render()

        state = next_state

        if done:
            state = env.reset()

    env.close()


def plot_render(env: gym.Env, agent: Union[DQN, PPO], steps: int = 3000) -> None:
    """Watch a plot representation of an agent in a given environment."""
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))  # only call this once

    for _ in range(steps):
        img.set_data(env.render(mode='rgb_array'))  # just update the data
        display.display(plt.gcf())
        display.clear_output(wait=True)

        if type(agent) == DQN:
            action = agent.act(state, 0.01)  # Generate an action
        else:  # PPO
            action = agent.act(state)
        next_state, reward, done, info = agent.env.step(action)  # Take an action

        state = next_state

        if done:
            state = env.reset()

    env.close()
