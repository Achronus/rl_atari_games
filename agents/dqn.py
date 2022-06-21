from collections import namedtuple
import numpy as np
import random
from datetime import datetime

from agents._agent import Agent
from core.buffer import ReplayBuffer, PrioritizedReplayBuffer
from core.parameters import AgentParameters, ModelParameters, BufferParameters, Experience
from core.env_details import EnvDetails
from utils.helper import number_to_num_letter, normalize, to_tensor, timer, timer_string
from utils.logger import DQNLogger, RDQNLogger

import torch
from torch.nn.utils import clip_grad_norm_

DQNExperience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class DQN(Agent):
    """
    A basic Deep Q-Network that uses an experience replay buffer and fixed Q-targets.

    Parameters:
        env_details (EnvDetails) - a class containing parameters for the environment
        model_params (ModelParameters) - a data class containing model specific parameters
        params (AgentParameters) - a data class containing DQN specific parameters
        seed (int) - an integer for recreating results
    """
    def __init__(self, env_details: EnvDetails, model_params: ModelParameters,
                 params: AgentParameters, seed: int) -> None:
        self.logger = DQNLogger()
        super().__init__(env_details, params, seed, self.logger)

        self.env = env_details.make_env('dqn')
        self.action_size = env_details.n_actions

        self.memory = ReplayBuffer(params.buffer_size, params.batch_size, self.device, seed)
        self.local_network = model_params.network.to(self.device)
        self.target_network = model_params.network.to(self.device)  # Fixed target network

        self.optimizer = model_params.optimizer
        self.loss = model_params.loss_metric

        self.timestep = 0

    def step(self, experience: DQNExperience) -> None:
        """Perform a learning step, if there are enough samples in memory."""
        # Store experience in memory
        self.memory.add(experience)

        # Learning every few update_steps
        self.timestep = (self.timestep + 1) % self.params.update_steps

        # Perform learning
        if self.timestep == 0 and len(self.memory) > self.params.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state: torch.Tensor, epsilon: float) -> int:
        """
        Returns an action for a given state based on an epsilon greedy policy.

        Parameters:
            state (torch.Tensor) - current state
            epsilon (float) - current epsilon
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

    def learn(self, experiences: tuple[torch.Tensor, ...]) -> None:
        """Updates the network parameters."""
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q-value from target network
        q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q-targets for the current state
        q_targets = rewards + (self.params.gamma * q_targets_next * (1 - dones))

        # Get expected Q-values from local network
        q_preds = self.local_network(states).gather(1, actions)

        # Compute and minimize loss
        loss = self.loss(q_preds, q_targets.to(torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log details
        self.log_data(
                q_targets_next=q_targets_next,
                q_targets=q_targets,
                q_preds=q_preds,
                train_losses=loss
        )

        # Update target network
        self.__soft_update()

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

        Parameters:
            num_episodes (int) - the number of iterations to train the agent on
            print_every (int) - the number of episodes before outputting information
            save_count (int) - the number of episodes before saving the model
        """
        # Set initial epsilon
        eps = self.params.eps_start

        # Output info to console
        buffer_idx, buffer_letter = number_to_num_letter(self.memory.buffer_size)
        timesteps_idx, timesteps_letter = number_to_num_letter(self.params.max_timesteps)
        self._initial_output(num_episodes, f'Buffer size: {int(buffer_idx)}{buffer_letter.lower()}, '
                                           f'batch size: {self.memory.batch_size}, '
                                           f'max timesteps: {int(timesteps_idx)}{timesteps_letter.lower()}, '
                                           f'num network updates: {self.params.update_steps}.')
        # Iterate over episodes
        for i_episode in range(1, num_episodes+1):
            # Initialize state and score
            score = 0
            state = self.env.reset()

            # Iterate over timesteps
            for t in range(self.params.max_timesteps):
                state = normalize(to_tensor(state)).to(self.device)
                action = self.act(state, eps)  # Generate an action
                next_state, reward, done, info = self.env.step(action)  # Take an action

                # Perform learning
                self.step(DQNExperience(state.cpu(), action, reward, normalize(next_state), done))

                # Update state and score
                state = next_state
                score += reward

                # Log actions and environment info
                self.log_data(actions=action, env_info=info)

                # Check if finished
                if done:
                    break

            # Log metrics
            self.log_data(ep_scores=score, epsilons=eps)

            # Decrease epsilon
            eps = max(self.params.eps_end, self.params.eps_decay * eps)

            # Display output and save model
            self.__output_progress(num_episodes, i_episode, print_every)
            self._save_model_condition(i_episode, save_count,
                                       filename=f'dqn_batch{self.memory.batch_size}',
                                       extra_data={
                                           'local_network': self.local_network.state_dict(),
                                           'target_network': self.target_network.state_dict(),
                                           'optimizer': self.optimizer,
                                           'loss_metric': self.loss
                                       })
        print(f"Training complete. Access metrics from 'logger' attribute.")

    def __output_progress(self, num_episodes: int, i_episode: int, print_every: int) -> None:
        """Provides a progress update on the model's training to the console."""
        first_episode = i_episode == 1
        last_episode = i_episode == num_episodes+1

        if first_episode or last_episode or i_episode % print_every == 0:
            ep_idx, ep_letter = number_to_num_letter(i_episode)  # 1000 -> 1K
            ep_total_idx, ep_total_letter = number_to_num_letter(num_episodes)

            print(f'({int(ep_idx)}{ep_letter}/{int(ep_total_idx)}{ep_total_letter}) ', end='')
            print(f'Episode Score: {int(self.logger.ep_scores[i_episode-1])}, '
                  f'Train Loss: {self.logger.train_losses[i_episode-1]:.5f}')


class RainbowDQN(Agent):
    """A Rainbow Deep Q-Network that uses six extensions a-top a traditional DQN.

    Parameters:
        env_details (EnvDetails) - a class containing parameters for the environment
        model_params (ModelParameters) - a data class containing model specific parameters
        params (AgentParameters) - a data class containing Rainbow DQN specific parameters
        buffer_params (BufferParameters) - a class containing parameters for the buffer
        seed (int) - an integer for recreating results
    """
    def __init__(self, env_details: EnvDetails, model_params: ModelParameters,
                 params: AgentParameters, buffer_params: BufferParameters, seed: int) -> None:
        self.logger = RDQNLogger()
        super().__init__(env_details, params, seed, self.logger)

        self.env = env_details.make_env('dqn')
        self.action_size = env_details.n_actions
        self.batch_size = buffer_params.batch_size
        self.buffer_params = buffer_params

        self.buffer = PrioritizedReplayBuffer(buffer_params, params.n_steps, env_details.stack_size,
                                              self.device, self.logger)
        self.local_network = model_params.network.to(self.device)
        self.target_network = model_params.network.to(self.device)

        self.optimizer = model_params.optimizer
        self.loss = model_params.loss_metric

        self.z_delta = (params.v_max - params.v_min) / (params.n_atoms - 1)
        self.z_support = torch.linspace(params.v_min, params.v_max, params.n_atoms).to(self.device)
        self.discount_scaling = torch.Tensor([self.params.gamma ** i for i in range(self.params.n_steps)],)
        self.priority_weight = buffer_params.priority_weight
        self.priority_weight_increase = 0.

        self.save_batch_time = datetime.now()  # init

    def act(self, state: torch.Tensor) -> int:
        """Returns an action for a given state using the local network."""
        with torch.no_grad():
            atom_action_probs = self.local_network.forward(state)
            action_probs = (atom_action_probs * self.z_support).sum(dim=2)  # shape -> batch_size, n_actions
            action = torch.argmax(action_probs).item()
        return action

    def learn(self) -> None:
        """Perform agent learning by updating the network parameters."""
        # Get samples from buffer
        samples = self.buffer.sample()

        # Calculate N-step returns
        returns = torch.matmul(samples['rewards'].cpu(), self.discount_scaling)

        # Compute Double-Q probabilities and values
        with torch.no_grad():
            double_q_probs = self.compute_double_q_probs(samples['next_states'].to(self.device))
            double_q = self.compute_double_q(samples, returns, double_q_probs)

        # Compute importance-sampling weights and log action probabilities
        weights = self.buffer.importance_sampling(samples['priorities'])  # shape -> [batch_size]
        log_action_probs = torch.log(self.local_network(samples['states'].to(self.device)))
        action_probs = log_action_probs[range(self.batch_size), samples['actions']]

        # Compute and minimize loss
        loss = -torch.sum(double_q * action_probs.cpu(), dim=1).to(self.device)  # Cross-entropy
        loss = (weights * loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.local_network.parameters(), self.params.clip_grad)
        self.optimizer.step()

        # Log details
        self.log_data(
            returns=returns.cpu(),
            actions=samples['actions'].detach().cpu(),
            double_q_values=double_q.cpu(),
            double_q_probs=double_q_probs.cpu(),
            train_losses=loss.item()
        )

        # Update buffer priorities
        self.buffer.update_priorities(samples['priority_indices'], loss.detach())

    def compute_double_q_probs(self, next_states: torch.Tensor) -> torch.Tensor:
        """Computes the Double-Q probabilities for the best actions obtained from the local network."""
        # Compute N-step next state probabilities
        probs = self.local_network.forward(next_states)  # Local net: next action probabilities
        probs_dist = self.z_support.expand_as(probs) * probs  # Fit probs in range of atoms [min, max]
        best_actions_indices = probs_dist.sum(2).argmax(1)  # Perform action selection

        self.target_network.sample_noise(self.device)  # Sample new target noise
        target_probs = self.target_network.forward(next_states)  # Target net: next action probabilities

        # Calculate Double-Q probabilities for best actions, shape -> [batch_size, n_atoms]
        return target_probs[range(self.batch_size), best_actions_indices].cpu()

    def compute_double_q(self, samples: dict, returns: torch.Tensor, double_q_probs: torch.Tensor) -> torch.Tensor:
        """Performs the categorical DQN operations to compute Double-Q values."""
        # Compute Tz (Bellman operator T applied to z) - Categorical DQN
        support = self.z_support.unsqueeze(0).cpu()
        tz = returns.unsqueeze(1) + samples['dones'] * (self.params.gamma ** self.params.n_steps) * support
        tz = tz.clamp(min=self.params.v_min, max=self.params.v_max)  # Limit values between atom [min, max]

        # Compute L2 projection of Tz onto z support
        projection = (tz - self.params.v_min) / self.z_delta
        lower_bound = projection.floor().to(torch.long)
        upper_bound = projection.ceil().to(torch.long)

        # Fix disappearing probabilities (due to integers) when lb = projection = ub
        lower_bound[(upper_bound > 0) * (lower_bound == upper_bound)] -= 1
        upper_bound[(lower_bound < (self.params.n_atoms - 1)) * (lower_bound == upper_bound)] += 1

        # Distribute probability of Tz
        double_q = samples['states'].new_zeros(self.batch_size, self.params.n_atoms)  # 1D array
        offset = torch.linspace(0, ((self.batch_size - 1) * self.params.n_atoms), self.batch_size)
        offset = offset.unsqueeze(1).expand(self.batch_size, self.params.n_atoms).to(samples['actions'])

        # Add flattened bounded offsets to Double-Q probabilities
        lb, ub = (lower_bound + offset).flatten(), (upper_bound + offset).flatten()
        double_lb = (double_q_probs * (upper_bound.float() - projection)).flatten()
        double_ub = (double_q_probs * (projection - lower_bound.float())).flatten()
        double_q.flatten().index_add_(dim=0, index=lb, source=double_lb)  # Add lower bounds
        double_q.flatten().index_add_(dim=0, index=ub, source=double_ub)  # Add upper bounds
        return double_q

    def train(self, num_episodes: int, print_every: int = 100, save_count: int = 1000) -> None:
        """
        Train the agent.

        Parameters:
            num_episodes (int) - the number of iterations to train the agent on
            print_every (int) - the number of episodes before outputting information
            save_count (int) - the number of episodes before saving the model
        """
        # Output info to console
        buffer_idx, buffer_letter = number_to_num_letter(self.buffer.capacity)
        timesteps_idx, timesteps_letter = number_to_num_letter(self.params.max_timesteps)
        self._initial_output(num_episodes, f'Buffer size: {int(buffer_idx)}{buffer_letter.lower()}, '
                                           f'batch size: {self.buffer.batch_size}, '
                                           f'max timesteps: {int(timesteps_idx)}{timesteps_letter.lower()}, '
                                           f'num network updates: {self.params.update_steps}, '
                                           f'replay period: {self.params.replay_period}.')

        # Time training
        with timer('Total time taken:'):
            self.save_batch_time = datetime.now()  # print_every start time
            # Iterate over episodes
            for i_episode in range(1, num_episodes + 1):
                # Initialize priority weight increase per episode
                self.priority_weight_increase = min(i_episode / num_episodes, 1)
                state = self.env.reset()  # Initialize state
                score = 0.
                env_info = []

                # Iterate over timesteps
                for timestep in range(self.params.max_timesteps):
                    state = normalize(to_tensor(state)).to(self.device)
                    action = self.act(state.unsqueeze(0))  # Generate an action
                    next_state, reward, done, info = self.env.step(action)  # Take an action
                    score += reward
                    reward = max(min(reward, self.params.reward_clip), -self.params.reward_clip)  # clip reward

                    # Add transition to buffer (storing only last state frame)
                    transition = Experience(state[-1].cpu(), action, reward, not done)
                    self.buffer.add(transition)

                    # Start learning after replay period is reached
                    if timestep >= self.params.replay_period and len(self.buffer) > self.batch_size:
                        # self.__anneal_weights()

                        # Learn every few timesteps
                        if timestep % self.params.learn_frequency == 0:
                            self.learn()

                    # Update target network every few timesteps
                    if timestep % self.params.update_steps == 0:
                        self.__soft_update_target_network()

                    # Sample new noise every replay period
                    if timestep % self.params.replay_period == 0:
                        self.local_network.sample_noise(self.device)

                    # Update state
                    state = next_state
                    env_info.append(info)  # add info to list

                    # Check if finished
                    if done:
                        break

                # Add items to logger
                self.log_data(ep_scores=score, env_info=env_info)

                # Display output and save model
                self.__output_progress(num_episodes, i_episode, print_every)
                self._save_model_condition(i_episode, save_count,
                                           filename=f'rainbow_batch{self.buffer.batch_size}_'
                                                    f'buffer_size{self.buffer.capacity}',
                                           extra_data={
                                               'local_network': self.local_network.state_dict(),
                                               'target_network': self.target_network.state_dict(),
                                               'optimizer': self.optimizer,
                                               'buffer_params': self.buffer_params
                                           })
            print(f"Training complete. Access metrics from 'logger' attribute.", end=' ')

    def __anneal_weights(self) -> None:
        """Anneal importance sampling weight (β) to 1 at the start of learning."""
        self.buffer.priority_weight = min(self.buffer.priority_weight + self.priority_weight_increase, 1)

    def __soft_update_target_network(self) -> None:
        """
        Performs a soft update of the target networks parameters.
        Formula: θ_target = Τ * θ_local + (1 - Τ) * θ_target
        """
        for target, local in zip(self.target_network.parameters(), self.local_network.parameters()):
            target.detach().copy_(self.params.tau * local.detach() + (1.0 - self.params.tau) * target.detach())

    def __output_progress(self, num_episodes: int, i_episode: int, print_every: int) -> None:
        """Provides a progress update on the model's training to the console."""
        first_episode = i_episode == 1
        last_episode = i_episode == num_episodes

        if first_episode or last_episode or i_episode % print_every == 0:
            ep_idx, ep_letter = number_to_num_letter(i_episode)  # 1000 -> 1K
            ep_total_idx, ep_total_letter = number_to_num_letter(num_episodes)
            time_taken = (datetime.now() - self.save_batch_time)

            print(f'({int(ep_idx)}{ep_letter}/{int(ep_total_idx)}{ep_total_letter})  ', end='')
            print(f'Episode Score: {int(self.logger.ep_scores[i_episode-1])},  ',
                  f'Train Loss: {self.logger.train_losses[i_episode-1]:.5f},  ', end='')
            print(timer_string(time_taken, 'Time taken:'))
            self.save_batch_time = datetime.now()  # Reset
