import gym
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

from agents._agent import Agent

import torch


def video_render(env: gym.Env, agent: Agent, steps: int = 3000) -> None:
    """Watch a video representation of an agent in a given environment."""
    state = env.reset()

    for _ in range(steps):
        normalize_state = (1.0 / 255) * np.asarray(state)
        state = torch.from_numpy(normalize_state).to(agent.config.device)
        action_probs = agent.config.network.forward(state.unsqueeze(0))[0]
        action = torch.distributions.Categorical(action_probs).sample().item()
        next_state, reward, done, _ = env.step(action)
        env.render()

        state = next_state

        if done:
            state = env.reset()

    env.close()


def plot_render(env: gym.Env, agent: Agent, steps: int = 3000) -> None:
    """Watch a plot representation of an agent in a given environment."""
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))  # only call this once

    for _ in range(steps):
        img.set_data(env.render(mode='rgb_array'))  # just update the data
        display.display(plt.gcf())
        display.clear_output(wait=True)

        normalize_state = (1.0 / 255) * np.asarray(state)
        state = torch.from_numpy(normalize_state).to(agent.config.device)
        action_probs = agent.config.network.forward(state.unsqueeze(0))[0]
        action = torch.distributions.Categorical(action_probs).sample().item()
        next_state, reward, done, _ = env.step(action)

        state = next_state

        if done:
            state = env.reset()

    env.close()
