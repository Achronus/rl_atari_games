import time
import numpy as np
from datetime import datetime
from collections import Counter

from agents._agent import Agent
from core.buffer import RolloutBuffer
from core.enums import ValidIMMethods
from core.env_details import EnvDetails
from core.parameters import ModelParameters, PPOParameters
from intrinsic.parameters import IMExperience
from utils.helper import to_tensor, normalize, number_to_num_letter, timer, timer_string
from utils.logger import PPOLogger

import torch
import torch.nn as nn
from supersuit import gym_vec_env_v0


class PPO(Agent):
    """
    A basic Proximal Policy Optimization (PPO) algorithm that follows the clip variant.

    :param env_details (EnvDetails) - a class containing parameters for the environment
    :param model_params (ModelParameters) - a data class containing model specific parameters
    :param params (PPOParameters) - a data class containing PPO specific parameters
    :param device (str) - name of CUDA device ('cpu' or 'cuda:0')
    :param seed (int) - an integer for recreating results
    :param im_type (tuple[str, IMController]) - the type of intrinsic motivation to use with its controller
    """
    def __init__(self, env_details: EnvDetails, model_params: ModelParameters,
                 params: PPOParameters, device: str, seed: int, im_type: tuple = None) -> None:
        self.logger = PPOLogger()
        super().__init__(env_details, params, device, seed, self.logger, im_type)

        self.envs = gym_vec_env_v0(env_details.make_env('ppo'), num_envs=params.num_envs, multiprocessing=True)
        self.buffer = RolloutBuffer(params.rollout_size, params.num_envs,
                                    env_details.input_shape, env_details.action_space.shape)

        self.network = model_params.network.to(self.device)

        self.optimizer = model_params.optimizer
        self.loss = model_params.loss_metric
        self.num_envs = params.num_envs

        self.total_timesteps: int = 0
        self.start_time = time.time()
        self.batch_size = int(params.num_envs * params.rollout_size)
        self.mini_batch_size = int(self.batch_size // params.num_mini_batches)

        self.save_batch_time = datetime.now()  # init

    @staticmethod
    def act(action_probs: torch.Tensor, action: torch.Tensor = None) -> dict:
        """
        Returns a dictionary containing the action, log probability
        and entropy for a given set of action probabilities.

        :param action_probs (torch.Tensor) - a tensor of action probabilities
        :param action (torch.Tensor) - existing action values, optional
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
        samples = self.buffer.sample(['states', 'dones', 'rewards', 'state_values', 'actions'])
        rtgs = torch.zeros_like(samples.rewards)  # Returns (rewards-to-go)

        # Get next_state_value
        with torch.no_grad():
            next_state = self.encode_state(samples.states[-1].squeeze(0).to(self.device))
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
        state, info = self.envs.reset()
        rewards = []

        # Iterate over rollout size
        for i_rollout in range(self.params.rollout_size):
            self.total_timesteps += 1 * self.num_envs  # timesteps so far
            state = normalize(to_tensor(state)).to(self.device)

            with torch.no_grad():
                action_probs, state_values = self.network.forward(self.encode_state(state))

            preds = self.act(action_probs)  # Get an action
            next_state, reward, done, _, info = self.envs.step(preds['action'].numpy())  # Take an action
            rewards.append(reward)
            reward = np.clip(reward, a_min=-1, a_max=1)  # clip reward

            # Add rollout to buffer
            self.add_to_buffer(
                step=i_rollout,
                states=state.cpu(),
                next_states=to_tensor(next_state),
                actions=preds['action'],
                rewards=torch.Tensor(reward),
                dones=torch.Tensor(done),
                log_probs=preds['log_prob'],
                state_values=state_values.flatten().cpu()
            )

        # Log info
        rewards = np.stack(rewards)
        self.log_data(avg_rewards=rewards.mean(axis=1).mean().item())

    def learn(self) -> None:
        """Performs agent learning."""
        kls, actions = [], []
        policy_losses, value_losses, entropy_losses, total_losses = [], [], [], []
        im_losses = []

        # Calculate advantages
        rtgs, advantages = self.compute_rtgs_and_advantages()

        # Flatten into batches
        rtgs, advantages = rtgs.reshape(-1), advantages.reshape(-1)

        # Get rollout batches from buffer
        data_batch = self.buffer.sample_batch(['states', 'actions', 'log_probs', 'state_values', 'next_states'])
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
                states = self.encode_state(data_batch.states[mini_batch_indices].to(self.device))
                action_probs, new_state_values = self.network.forward(states)
                y_preds = self.act(action_probs, data_batch.actions[mini_batch_indices].to(self.device))

                # Calculate losses
                log_ratio = y_preds['log_prob'] - data_batch.log_probs[mini_batch_indices]
                ratio = torch.exp(log_ratio)
                approx_kl = (-log_ratio).mean()  # Debugging variable
                mini_returns, mini_state_values = rtgs[mini_batch_indices], data_batch.state_values[mini_batch_indices]

                # Add intrinsic reward
                if self.im_method is not None:
                    im_exp = IMExperience(
                        states,
                        self.encode_state(data_batch.next_states[mini_batch_indices].to(self.device)),
                        data_batch.actions[mini_batch_indices].to(torch.long).to(self.device)
                    )
                    im_return = normalize(self.im_method.module.compute_return(im_exp)).squeeze().cpu()
                    mini_returns += im_return

                policy_loss = self.clip_surrogate(ratio, mini_batch_advantages).mean()
                value_loss = self.clipped_value_loss(new_state_values.cpu(), mini_returns, mini_state_values)
                entropy_loss = y_preds['entropy'].mean()  # Encourages agent exploration
                loss = policy_loss - self.params.entropy_coef * entropy_loss + value_loss * self.params.value_loss_coef

                # Add intrinsic loss
                if self.im_method is not None:
                    loss, im_loss = self.im_method.module.compute_loss(im_exp, loss)
                    im_losses.append(im_loss)

                # Back-propagate loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.params.clip_grad)
                self.optimizer.step()

                # Add metrics to lists
                kls.append(approx_kl.item())
                actions.append(Counter(y_preds['action'].to(torch.int32).tolist()))
                policy_losses.append(policy_loss.detach().item())
                value_losses.append(value_loss.detach().item())
                entropy_losses.append(entropy_loss.detach().item())
                total_losses.append(loss.detach().item())

        # Add episodic info to the logger
        self.log_data(
            approx_kl=self._calc_mean(kls),
            policy_losses=self._calc_mean(policy_losses),
            value_losses=self._calc_mean(value_losses),
            entropy_losses=self._calc_mean(entropy_losses),
            total_losses=self._calc_mean(total_losses),
            avg_returns=self._calc_mean(rtgs),
            actions=self._count_actions(actions)
        )

        # Add intrinsic loss to logger if available
        if self.im_method is not None:
            im_loss = im_losses[-1] if self.im_type == ValidIMMethods.EMPOWERMENT.value else im_losses[-1].item()
            self.log_data(intrinsic_losses=im_loss)

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

    def train(self, num_episodes: int, print_every: int = 100, save_count: int = 1000,
              custom_ep_start: int = 0) -> None:
        """
        Train the agent.

        :param num_episodes (int) - the number of iterations to train the agent on
        :param print_every (int) - the number of episodes before outputting information
        :param save_count (int) - the number of episodes before saving the model
        :param custom_ep_start (int) - (optional) a custom parameter for the episode start number.
               Used exclusively for file save names. Useful when retraining models using retrain_model()
        """
        num_episodes = self.params.rollout_size * self.num_envs * num_episodes
        num_updates = num_episodes // self.batch_size  # Training iterations
        assert not num_updates == 0, f"'num_episodes' must be larger than the 'batch_size': {self.batch_size}!"

        # Output info to console
        self._initial_output(num_episodes,
                             f'Surrogate clipping size: {self.params.loss_clip}, '
                             f'rollout size: {self.params.rollout_size}, '
                             f'num environments: {self.params.num_envs}, '
                             f'num network updates: {self.params.update_steps}, '
                             f'batch size: {self.batch_size}, '
                             f'training iterations: {num_updates}, '
                             f'intrinsic method: {self.im_type}.')

        # Time training
        with timer('Total time taken:'):
            self.save_batch_time = datetime.now()  # print_every start time
            # Iterate over training iterations
            for i_episode in range(1+custom_ep_start, num_updates+1+custom_ep_start):
                # Create rollouts and store in buffer
                self.buffer.reset()  # Empty each episode
                self.generate_rollouts()

                # Perform learning
                self.learn()

                # Display output and save model
                model_name = f'ppo-{self.im_type[:3]}' if self.im_type is not None else 'ppo'
                self.__output_progress(num_updates+custom_ep_start, i_episode, print_every, custom_ep_start)
                extra_data = {
                    'network': self.network.state_dict(),
                    'network_type': self.network.__class__.__name__,
                    'optimizer': self.optimizer,
                    'loss_metric': self.loss,
                    'im_type': self.im_type
                }
                if self.im_type == ValidIMMethods.EMPOWERMENT.value:
                    im_model_data = {
                        'encoder': self.im_method.model.encoder.state_dict(),
                        'source_net': self.im_method.model.source_net.state_dict(),
                        'forward_net': self.im_method.model.forward_net.state_dict(),
                        'source_target': self.im_method.model.source_target.state_dict(),
                        'forward_target': self.im_method.model.forward_target.state_dict()
                    }
                    extra_data = {**extra_data, **im_model_data}

                self._save_model_condition(i_episode, save_count,
                                           filename=f'{model_name}_rollout{self.params.rollout_size}'
                                                    f'_envs{self.params.num_envs}',
                                           extra_data=extra_data)
            print(f"Training complete. Access metrics from 'logger' attribute.", end=' ')

    def __output_progress(self, num_episodes: int, i_episode: int, print_every: int, custom_ep_start: int = 0) -> None:
        """Provides a progress update on the model's training to the console."""
        first_episode = i_episode == 1+custom_ep_start
        last_episode = i_episode == num_episodes

        if first_episode or last_episode or i_episode % print_every == 0:
            ep_idx, ep_letter = number_to_num_letter(i_episode)  # 1000 -> 1K
            ep_total_idx, ep_total_letter = number_to_num_letter(num_episodes)
            time_taken = (datetime.now() - self.save_batch_time)

            print(f'({ep_idx:.1f}{ep_letter}/{int(ep_total_idx)}{ep_total_letter}) ', end='')
            print(f'Episode Score: {self.logger.avg_rewards[i_episode-1]:.2f},  '
                  f'Episodic Return: {self.logger.avg_returns[i_episode-1]:.2f},  '
                  f'Approx KL: {self.logger.approx_kl[i_episode-1]:.3f},  '
                  f'Total Loss: {self.logger.total_losses[i_episode-1]:.3f},  '
                  f'Policy Loss: {self.logger.policy_losses[i_episode-1]:.3f},  '
                  f'Value Loss: {self.logger.value_losses[i_episode-1]:.3f},  '
                  f'Entropy Loss: {self.logger.entropy_losses[i_episode-1]:.3f},  ', end='')

            if self.im_method is not None:
                intrinsic_losses = self.logger.intrinsic_losses[i_episode - 1]
                if isinstance(intrinsic_losses, list):
                    print(f'Source Loss: {intrinsic_losses[0]:.5f},  Forward Loss: {intrinsic_losses[1]:.5f},  ', end='')
                else:
                    print(f'{self.im_type.title()} Loss: {intrinsic_losses:.5f},  ', end='')

            print(timer_string(time_taken, 'Time taken:'))
            self.save_batch_time = datetime.now()  # Reset

    def add_to_buffer(self, step: int, **kwargs) -> None:
        """Adds data to the buffer."""
        self.buffer.add(step, **kwargs)
