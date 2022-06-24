from collections import Counter
from datetime import datetime

from agents._agent import Agent
from core.buffer import PrioritizedReplayBuffer
from core.parameters import AgentParameters, ModelParameters, BufferParameters, Experience
from core.env_details import EnvDetails
from utils.helper import number_to_num_letter, normalize, to_tensor, timer, timer_string
from utils.logger import RDQNLogger

import torch
import torch.nn as nn


class RainbowDQN(Agent):
    """A Rainbow Deep Q-Network that uses six extensions a-top a traditional DQN.

    Parameters:
        env_details (EnvDetails) - a class containing parameters for the environment
        model_params (ModelParameters) - a data class containing model specific parameters
        params (AgentParameters) - a data class containing Rainbow DQN specific parameters
        buffer_params (BufferParameters) - a class containing parameters for the buffer
        device (str) - name of CUDA device ('cpu' or 'cuda:0')
        seed (int) - an integer for recreating results
    """
    def __init__(self, env_details: EnvDetails, model_params: ModelParameters,
                 params: AgentParameters, buffer_params: BufferParameters, device: str, seed: int) -> None:
        self.logger = RDQNLogger()
        super().__init__(env_details, params, device, seed, self.logger)

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

    def learn(self) -> tuple:
        """Perform agent learning by updating the network parameters.
        Returns the steps average return and training loss."""
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
        nn.utils.clip_grad_norm_(self.local_network.parameters(), self.params.clip_grad)
        self.optimizer.step()

        # Log actions
        self.log_data(
            actions=Counter(samples['actions'].detach().cpu().tolist())
        )

        # Update buffer priorities
        self.buffer.update_priorities(samples['priority_indices'], loss.detach())
        avg_return = returns.mean().item()
        return avg_return, loss.item()

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
                avg_returns, train_losses = [], []

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

                        # Learn every few timesteps
                        if timestep % self.params.learn_frequency == 0:
                            avg_return, train_loss = self.learn()

                            # Add items to lists
                            avg_returns.append(avg_return)
                            train_losses.append(train_loss)

                    # Update target network every few timesteps
                    if timestep % self.params.update_steps == 0:
                        self.__soft_update_target_network()

                    # Sample new noise every replay period
                    if timestep % self.params.replay_period == 0:
                        self.local_network.sample_noise(self.device)

                    # Update state
                    state = next_state

                    # Check if finished
                    if done:
                        break

                # Add episode data to logger
                self.log_data(
                    ep_scores=score,
                    avg_returns=self._calc_mean(avg_returns),
                    train_losses=self._calc_mean(train_losses)
                )

                # Display output and save model
                self.__output_progress(num_episodes, i_episode, print_every)
                self._save_model_condition(i_episode, save_count,
                                           filename=f'rainbow_batch{self.buffer.batch_size}_'
                                                    f'buffer{self.buffer.capacity}',
                                           extra_data={
                                               'local_network': self.local_network.state_dict(),
                                               'target_network': self.target_network.state_dict(),
                                               'optimizer': self.optimizer,
                                               'buffer_params': self.buffer_params
                                           })
            print(f"Training complete. Access metrics from 'logger' attribute.", end=' ')

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
