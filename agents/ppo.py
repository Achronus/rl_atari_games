import numpy as np

from agents._agent import Agent
from core.buffer import RolloutBuffer
from core.env_details import EnvDetails
from core.parameters import PPOModelParameters, PPOParameters
from utils.logger import PPOLogger

import torch


class PPO(Agent):
    """
    A basic Proximal Policy Optimization (PPO) algorithm that follows the clip variant.
    Pseudocode shown in OpenAI's Spinning Up documentation (https://spinningup.openai.com/en/latest/algorithms/ppo.html#id7).

    Parameters:
        env_details (EnvDetails) - a class containing parameters for the environment
        model_params (PPOModelParameters) - a data class containing model specific parameters
        params (PPOParameters) - a data class containing PPO specific parameters
        seed (int) - an integer for recreating results
    """
    def __init__(self, env_details: EnvDetails, model_params: PPOModelParameters,
                 params: PPOParameters, seed: int) -> None:
        self.logger = PPOLogger()
        super().__init__(env_details, params, seed, self.logger)

        self.envs = [env_details.env for i in range(params.num_agents)]
        self.buffer = RolloutBuffer(params.rollout_size, params.num_agents,
                                    env_details.input_shape, env_details.n_actions)

        self.actor = model_params.actor.to(self.device)
        self.critic = model_params.critic.to(self.device)

        self.actor_optimizer = model_params.actor_optimizer
        self.critic_optimizer = model_params.critic_optimizer
        self.loss = model_params.loss_metric

    @staticmethod
    def act(action_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns an action and its log probability for a given set of action probabilities."""
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().detach(), log_prob.cpu().detach()

    def generate_rollouts(self) -> list:
        """Generates a set of rollouts and stores them in the buffer."""
        pass

    def learn(self) -> None:
        """Performs agent learning."""
        pass

    def train(self, num_episodes: int, print_every: int = 100, save_count: int = 1000) -> None:
        """
        Train the agent.

        Parameters:
            num_episodes (int) - the number of iterations to train the agent on
            print_every (int) - the number of episodes before outputting information
            save_count (int) - the number of episodes before saving the model
        """
        pass

    def add_to_buffer(self, **kwargs) -> None:
        """Adds data to the buffer."""
        self.buffer.add(**kwargs)
