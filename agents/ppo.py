import time
import numpy as np
from datetime import datetime

from agents._agent import Agent
from core.buffer import RolloutBuffer
from core.env_details import EnvDetails
from core.parameters import ModelParameters, PPOParameters
from utils.helper import to_tensor, normalize, number_to_num_letter, timer, timer_string
from utils.logger import PPOLogger

import torch
import torch.nn as nn
from supersuit import gym_vec_env_v0


class PPO(Agent):
    """
    A basic Proximal Policy Optimization (PPO) algorithm that follows the clip variant.

    Parameters:
        env_details (EnvDetails) - a class containing parameters for the environment
        model_params (ModelParameters) - a data class containing model specific parameters
        params (PPOParameters) - a data class containing PPO specific parameters
        seed (int) - an integer for recreating results
    """
    def __init__(self, env_details: EnvDetails, model_params: ModelParameters,
                 params: PPOParameters, seed: int) -> None:
        self.logger = PPOLogger()
        super().__init__(env_details, params, seed, self.logger)

        self.envs = gym_vec_env_v0(env_details.make_env('ppo'), num_envs=params.num_agents, multiprocessing=True)
        self.buffer = RolloutBuffer(params.rollout_size, params.num_agents,
                                    env_details.input_shape, env_details.action_space.shape)

        self.network = model_params.network.to(self.device)

        self.optimizer = model_params.optimizer
        self.loss = model_params.loss_metric
        self.num_agents = params.num_agents

        self.total_timesteps: int = 0
        self.start_time = time.time()
        self.batch_size = int(params.num_agents * params.rollout_size)
        self.mini_batch_size = int(self.batch_size // params.num_mini_batches)

        self.save_batch_time = datetime.now()  # init

    @staticmethod
    def act(action_probs: torch.Tensor, action: torch.Tensor = None) -> dict:
        """
        Returns a dictionary containing the action, log probability
        and entropy for a given set of action probabilities.

        Parameters:
            action_probs (torch.Tensor) - a tensor of action probabilities
            action (torch.Tensor) - existing action values, optional
        """
        dist = torch.distributions.Categorical(action_probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return {
            'action': action.cpu(),
            'log_prob': log_prob.cpu(),
            'entropy': dist.entropy().cpu()
        }

    def compute_rtgs_and_advantages(self) -> tuple:
        """Computes advantages using rewards-to-go (rtgs/returns)."""
        # Get rollouts and initialize rtgs
        samples = self.buffer.sample(['states', 'dones', 'rewards', 'state_values'])
        rtgs = torch.zeros_like(samples.rewards)  # Returns (rewards-to-go)

        # Get next_state_value
        with torch.no_grad():
            next_state = samples.states[-1].squeeze(0).to(self.device)
            next_state_value = self.network.forward(next_state)[1].reshape(1, -1)

        # Iterate over rollouts backwards
        for i_rollout in reversed(range(self.params.rollout_size)):
            if i_rollout == self.params.rollout_size - 1:
                next_non_terminal = 1.0 - samples.dones[-1]
                next_return = next_state_value
            else:
                next_non_terminal = 1.0 - samples.dones[i_rollout + 1]
                next_return = rtgs[i_rollout + 1]

            # Compute returns
            rtgs[i_rollout] = samples.rewards[i_rollout] + self.params.gamma * next_non_terminal * next_return.cpu()

        # Compute advantages and normalize (1e-10 avoids dividing by 0)
        advantages = rtgs - samples.state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        return rtgs, advantages

    def generate_rollouts(self) -> None:
        """Generates a set of rollouts and stores them in the buffer."""
        # Initialize values for each episode
        state = self.envs.reset()
        env_info, rewards = [], []

        # Iterate over rollout size
        for i_rollout in range(self.params.rollout_size):
            self.total_timesteps += 1 * self.num_agents  # timesteps so far
            state = normalize(to_tensor(state)).to(self.device)

            with torch.no_grad():
                action_probs, state_values = self.network.forward(state)

            preds = self.act(action_probs)  # Get an action
            next_state, reward, done, info = self.envs.step(preds['action'].numpy())  # Take an action
            reward = np.clip(reward, a_min=-1, a_max=1)  # clip reward

            # Add rollout to buffer
            self.add_to_buffer(
                step=i_rollout,
                states=state.cpu(),
                actions=preds['action'],
                rewards=torch.Tensor(reward),
                dones=torch.Tensor(done),
                log_probs=preds['log_prob'],
                state_values=state_values.flatten().cpu()
            )

            # Add data to list
            rewards.append(reward)
            env_info.append(info)

        # Log info
        self.log_data(env_info=env_info, rewards=rewards)

    def learn(self) -> None:
        """Performs agent learning."""
        kls, predictions, log_ratios, ratios = [], [], [], []
        policy_losses, value_losses, entropy_losses, total_losses = [], [], [], []

        # Calculate advantages
        rtgs, advantages = self.compute_rtgs_and_advantages()

        # Flatten into batches
        rtgs, advantages = rtgs.reshape(-1), advantages.reshape(-1)

        # Get rollout batches from buffer
        data_batch = self.buffer.sample_batch(['states', 'actions', 'log_probs', 'state_values'])
        batch_indices = np.arange(self.batch_size)

        # Iterate over update steps
        for step in range(self.params.update_steps):
            np.random.shuffle(batch_indices)  # Shuffle batches

            # Iterate over each mini-batch
            for i_start in range(0, self.batch_size, self.mini_batch_size):
                # Get a minibatch
                batch_end = i_start + self.mini_batch_size
                mini_batch_indices = batch_indices[i_start:batch_end]
                mini_batch_advantages = advantages[mini_batch_indices]

                # Calculate network predictions
                states = data_batch.states[mini_batch_indices].to(self.device)
                action_probs, new_state_values = self.network.forward(states)
                y_preds = self.act(action_probs, data_batch.actions[mini_batch_indices].to(self.device))

                # Calculate losses
                log_ratio = y_preds['log_prob'] - data_batch.log_probs[mini_batch_indices]
                ratio = torch.exp(log_ratio)
                approx_kl = (-log_ratio).mean()  # Debugging variable
                mini_returns, mini_state_values = rtgs[mini_batch_indices], data_batch.state_values[mini_batch_indices]

                policy_loss = self.clip_surrogate(ratio, mini_batch_advantages).mean()
                value_loss = self.clipped_value_loss(new_state_values.cpu(), mini_returns, mini_state_values)
                entropy_loss = y_preds['entropy'].mean()  # Encourages agent exploration
                loss = policy_loss - self.params.entropy_coef * entropy_loss + value_loss * self.params.value_loss_coef

                # Back-propagate loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.params.clip_grad)
                self.optimizer.step()

                # Add metrics to lists
                kls.append(approx_kl.item())
                predictions.append(y_preds)
                log_ratios.append(log_ratio.detach())
                ratios.append(ratio.detach())
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropy_losses.append(entropy_loss.detach())
                total_losses.append(loss.detach())

        # Add episodic info to the logger
        self.log_data(approx_kl=to_tensor(kls), returns=rtgs.detach(), advantages=advantages.detach(),
                      predictions=predictions, log_ratios=log_ratios, ratios=ratios,
                      policy_losses=to_tensor(policy_losses), value_losses=to_tensor(value_losses),
                      entropy_losses=to_tensor(entropy_losses), total_losses=to_tensor(total_losses))

    def clipped_value_loss(self, new_state_value: torch.Tensor, batch_returns: torch.Tensor,
                           batch_state_values: torch.Tensor) -> torch.Tensor:
        """Computes the clipped value loss and returns it."""
        value_loss = (new_state_value - batch_returns) ** 2
        state_value_clipped = batch_state_values + self.__state_value_clip(new_state_value, batch_state_values)
        value_loss_clipped = (state_value_clipped - batch_returns) ** 2
        return 0.5 * torch.max(value_loss, value_loss_clipped).mean()

    def __state_value_clip(self, new_state_value: torch.Tensor, batch_state_values: torch.Tensor) -> torch.Tensor:
        """Clips the state value using torch.clamp between a range of [-loss_clip, +loss_clip]."""
        loss_clip = self.params.loss_clip
        return torch.clamp(new_state_value - batch_state_values, min=-loss_clip, max=loss_clip)

    def clip_surrogate(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """Performs the clipped surrogate function, returning the updated output."""
        clip = torch.clamp(ratio, min=1 - self.params.loss_clip, max=1 + self.params.loss_clip)
        return torch.min(advantages * ratio, advantages * clip)

    def train(self, num_episodes: int, print_every: int = 100, save_count: int = 1000) -> None:
        """
        Train the agent.

        Parameters:
            num_episodes (int) - the number of iterations to train the agent on
            print_every (int) - the number of episodes before outputting information
            save_count (int) - the number of episodes before saving the model
        """
        num_updates = num_episodes // self.batch_size  # Training iterations
        assert not num_updates == 0, f"'num_episodes' must be larger than the 'batch_size': {self.batch_size}!"

        # Output info to console
        self._initial_output(num_episodes,
                             f'Surrogate clipping size: {self.params.loss_clip}, '
                             f'rollout size: {self.params.rollout_size}, '
                             f'num agents: {self.params.num_agents}, '
                             f'num network updates: {self.params.update_steps}, '
                             f'batch size: {self.batch_size}, '
                             f'training iterations: {num_updates}.')

        # Time training
        with timer('Total time taken:'):
            self.save_batch_time = datetime.now()  # print_every start time
            # Iterate over training iterations
            for i_episode in range(1, num_updates+1):
                # Create rollouts and store in buffer
                self.buffer.reset()  # Empty each episode
                self.generate_rollouts()

                # Perform learning
                self.learn()

                # Display output and save model
                self.__output_progress(num_updates, i_episode, print_every)
                self._save_model_condition(i_episode, save_count,
                                           filename=f'ppo_rollout{self.params.rollout_size}'
                                                    f'_agents{self.params.num_agents}',
                                           extra_data={
                                               'network': self.network.state_dict(),
                                               'optimizer': self.optimizer,
                                               'loss_metric': self.loss
                                           })
            print(f"Training complete. Access metrics from 'logger' attribute.", end=' ')

    def __output_progress(self, num_episodes: int, i_episode: int, print_every: int) -> None:
        """Provides a progress update on the model's training to the console."""
        first_episode = i_episode == 1
        last_episode = i_episode == num_episodes

        if first_episode or last_episode or i_episode % print_every == 0:
            ep_idx, ep_letter = number_to_num_letter(i_episode)  # 1000 -> 1K
            ep_total_idx, ep_total_letter = number_to_num_letter(num_episodes)
            ep, data = i_episode-1, self.logger
            time_taken = (datetime.now() - self.save_batch_time)

            print(f'({int(ep_idx)}{ep_letter}/{int(ep_total_idx)}{ep_total_letter}) ', end='')
            print(f'Episodic Return: {self.__mean_val(data.returns, ep)},  '
                  f'Approx KL: {self.__mean_val(data.approx_kl, ep)},  '
                  f'Total Loss: {self.__mean_val(data.total_losses, ep)},  '
                  f'Policy Loss: {self.__mean_val(data.policy_losses, ep)},  '
                  f'Value Loss: {self.__mean_val(data.value_losses, ep)},  '
                  f'Entropy Loss: {self.__mean_val(data.entropy_losses, ep)},  ', end='')
            print(timer_string(time_taken, 'Time taken:'))
            self.save_batch_time = datetime.now()  # Reset

    @staticmethod
    def __mean_val(data: torch.Tensor, i_episode: int) -> str:
        """
        Calculates the mean for a given tensor of data at a corresponding episode and
        returns a string representation to 5 decimal places.
        """
        return f'{data[i_episode].mean().item():.5f}'

    def add_to_buffer(self, step: int, **kwargs) -> None:
        """Adds data to the buffer."""
        self.buffer.add(step, **kwargs)
