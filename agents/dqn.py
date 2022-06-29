from collections import namedtuple, Counter
from typing import Union
import numpy as np
import random
from datetime import datetime

from agents._agent import Agent
from core.buffer import ReplayBuffer
from core.parameters import AgentParameters, ModelParameters
from core.env_details import EnvDetails
from intrinsic.controller import IMController
from intrinsic.parameters import IMExperience
from utils.helper import number_to_num_letter, normalize, to_tensor, timer
from utils.logger import DQNLogger

import torch

DQNExperience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class DQN(Agent):
    """
    A basic Deep Q-Network that uses an experience replay buffer and fixed Q-targets.
    """
    def __init__(self, env_details: EnvDetails, model_params: ModelParameters,
                 params: AgentParameters, device: str, seed: int, im_type: tuple = None) -> None:
        """
        :param env_details (EnvDetails) - a class containing parameters for the environment
        :param model_params (ModelParameters) - a data class containing model specific parameters
        :param params (AgentParameters) - a data class containing DQN specific parameters
        :param device (str) - name of CUDA device ('cpu' or 'cuda:0')
        :param seed (int) - an integer for recreating results
        :param im_type (tuple[str, IMParameters]) - indicates the type of intrinsic motivation to use with its parameters
        """
        self.logger = DQNLogger()
        super().__init__(env_details, params, device, seed, self.logger)

        self.env = env_details.make_env('dqn')
        self.action_size = env_details.n_actions

        self.memory = ReplayBuffer(params.buffer_size, params.batch_size, self.device, seed)

        self.local_network = model_params.network.to(self.device)
        self.target_network = model_params.network.to(self.device)  # Fixed target network

        self.optimizer = model_params.optimizer
        self.loss = model_params.loss_metric

        self.im_type = None
        self.im_method = None

        if im_type is not None:
            self.im_type = im_type[0]
            self.im_method = IMController(im_type[0], im_type[1], self.device)

        self.timestep = 0
        self.save_batch_time = datetime.now()  # init

    def step(self, experience: DQNExperience) -> Union[float, None]:
        """Perform a learning step, if there are enough samples in memory.
        Returns the training loss and Q-value predictions."""
        # Store experience in memory
        self.memory.add(experience)

        # Learning every few update_steps
        self.timestep = (self.timestep + 1) % self.params.update_steps

        # Perform learning
        if self.timestep == 0 and len(self.memory) > self.params.batch_size:
            experiences = self.memory.sample()
            train_loss = self.learn(experiences)
            return train_loss
        return None

    def act(self, state: torch.Tensor, epsilon: float) -> int:
        """
        Returns an action for a given state based on an epsilon greedy policy.

        :param state (torch.Tensor) - current state
        :param epsilon (float) - current epsilon
        """
        state = state.unsqueeze(0)

        # Set to evaluation mode and get actions
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)

        # Set back to training mode
        self.local_network.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().numpy()).item()
        return random.choice(np.arange(self.action_size))

    def learn(self, experiences: tuple) -> float:
        """Updates the network parameters. Returns the training loss and average return."""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q-value from target network
        q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q-targets for the current state
        q_targets = rewards + (self.params.gamma * q_targets_next * (1 - dones))

        # Add intrinsic reward
        if self.im_method is not None:
            im_exp = IMExperience(states, next_states, actions)
            im_return = normalize(self.im_method.module.compute_return(im_exp))
            q_targets += im_return
            q_targets = q_targets.detach()

        # Get expected Q-values from local network
        q_preds = self.local_network(states).gather(1, actions)

        # Compute and minimize loss
        loss = self.loss(q_preds, q_targets.to(torch.float32))

        # Add intrinsic loss
        if self.im_method is not None:
            loss = self.im_method.module.compute_loss(im_exp, loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.__soft_update()

        return loss.item()

    def __soft_update(self) -> None:
        """
        Performs a soft update of the target networks parameters.
        Formula: θ_target = Τ * θ_local + (1 - Τ) * θ_target
        """
        for target, local in zip(self.target_network.parameters(), self.local_network.parameters()):
            target.detach().copy_(self.params.tau * local.detach() + (1.0 - self.params.tau) * target.detach())

    def train(self, num_episodes: int, print_every: int = 100, save_count: int = 1000) -> None:
        """
        Train the agent.

        :param num_episodes (int) - the number of iterations to train the agent on
        :param print_every (int) - the number of episodes before outputting information
        :param save_count (int) - the number of episodes before saving the model
        """
        # Set initial epsilon
        eps = self.params.eps_start

        # Output info to console
        buffer_idx, buffer_letter = number_to_num_letter(self.memory.buffer_size)
        timesteps_idx, timesteps_letter = number_to_num_letter(self.params.max_timesteps)
        self._initial_output(num_episodes, f'Buffer size: {int(buffer_idx)}{buffer_letter.lower()}, '
                                           f'batch size: {self.memory.batch_size}, '
                                           f'max timesteps: {int(timesteps_idx)}{timesteps_letter.lower()}, '
                                           f'num network updates: {self.params.update_steps}, '
                                           f'intrinsic method: {self.im_type}')

        # Time training
        with timer('Total time taken:'):
            self.save_batch_time = datetime.now()  # print_every start time
            # Iterate over episodes
            for i_episode in range(1, num_episodes+1):
                # Initialize state and score
                score = 0
                state = self.env.reset()
                actions, train_losses = [], []

                # Iterate over timesteps
                for t in range(self.params.max_timesteps):
                    state = normalize(to_tensor(state)).to(self.device)
                    action = self.act(state, eps)  # Generate an action
                    next_state, reward, done, info = self.env.step(action)  # Take an action

                    # Perform learning
                    exp = DQNExperience(state.cpu(), action, reward, normalize(next_state), done)
                    train_loss = self.step(exp)

                    # Update state and score
                    state = next_state
                    score += reward

                    # Check if finished
                    if done:
                        break

                    # Add items to list
                    actions.append(action)

                    if train_loss is not None:
                        train_losses.append(train_loss)

                # Log episodic metrics
                self.log_data(
                    ep_scores=score,
                    actions=Counter(actions),
                    train_losses=train_losses[-1]
                )

                # Decrease epsilon
                eps = max(self.params.eps_end, self.params.eps_decay * eps)

                # Display output and save model
                model_name = f'dqn{self.im_type}' if self.im_type is not None else 'dqn'
                self.__output_progress(num_episodes, i_episode, print_every)
                self._save_model_condition(i_episode, save_count,
                                           filename=f'{model_name}_batch{self.memory.batch_size}',
                                           extra_data={
                                               'local_network': self.local_network.state_dict(),
                                               'target_network': self.target_network.state_dict(),
                                               'optimizer': self.optimizer,
                                               'loss_metric': self.loss
                                           })
        print(f"Training complete. Access metrics from 'logger' attribute.", end=' ')

    def __output_progress(self, num_episodes: int, i_episode: int, print_every: int) -> None:
        """Provides a progress update on the model's training to the console."""
        first_episode = i_episode == 1
        last_episode = i_episode == num_episodes+1

        if first_episode or last_episode or i_episode % print_every == 0:
            ep_idx, ep_letter = number_to_num_letter(i_episode)  # 1000 -> 1K
            ep_total_idx, ep_total_letter = number_to_num_letter(num_episodes)

            print(f'({ep_idx:.1f}{ep_letter}/{int(ep_total_idx)}{ep_total_letter}) ', end='')
            print(f'Episode Score: {int(self.logger.ep_scores[i_episode-1])}, '
                  f'Train Loss: {self.logger.train_losses[i_episode-1]:.5f}')
