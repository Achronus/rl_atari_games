from collections import Counter
from datetime import datetime

from agents._agent import Agent
from core.buffer import PrioritizedReplayBuffer
from core.enums import ValidIMMethods
from core.parameters import AgentParameters, ModelParameters, BufferParameters, Experience
from core.env_details import EnvDetails
from intrinsic.parameters import IMExperience
from utils.helper import number_to_num_letter, normalize, to_tensor, timer, timer_string
from utils.logger import RDQNLogger

import torch
import torch.nn as nn


class RainbowDQN(Agent):
    """A Rainbow Deep Q-Network that uses six extensions a-top a traditional DQN.

    :param env_details (EnvDetails) - a class containing parameters for the environment
    :param model_params (ModelParameters) - a data class containing model specific parameters
    :param params (AgentParameters) - a data class containing Rainbow DQN specific parameters
    :param buffer_params (BufferParameters) - a class containing parameters for the buffer
    :param device (str) - name of CUDA device ('cpu' or 'cuda:0')
    :param seed (int) - an integer for recreating results
    :param im_type (tuple[str, IMController]) - the type of intrinsic motivation to use with its controller
    """
    def __init__(self, env_details: EnvDetails, model_params: ModelParameters, params: AgentParameters,
                 buffer_params: BufferParameters, device: str, seed: int, im_type: tuple = None) -> None:
        self.logger = RDQNLogger()
        super().__init__(env_details, params, device, seed, self.logger, im_type)

        self.env = env_details.make_env('rainbow')
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
            state = self.encode_state(state)
            atom_action_probs = self.local_network.forward(state)
            action_probs = (atom_action_probs * self.z_support).sum(dim=2)  # shape -> (batch_size, n_actions)
            action = torch.argmax(action_probs).item()
        return action

    def learn(self) -> tuple:
        """Perform agent learning by updating the network parameters.
        Returns the steps average return and training loss."""
        # Get samples from buffer
        samples = self.buffer.sample()
        im_loss = None

        # Encode states if using empowerment
        states = self.encode_state(samples['states'].to(self.device))
        next_states = self.encode_state(samples['next_states'].to(self.device))

        # Calculate N-step returns
        returns = torch.matmul(samples['rewards'].cpu(), self.discount_scaling)

        # Add intrinsic reward
        if self.im_method is not None:
            im_exp = IMExperience(states, next_states, samples['actions'].to(self.device))
            im_return = normalize(self.im_method.module.compute_return(im_exp)).squeeze().cpu()
            returns += im_return

        # Compute Double-Q probabilities and values
        with torch.no_grad():
            double_q_probs = self.compute_double_q_probs(next_states)
            double_q = self.compute_double_q(states.cpu(), samples, returns, double_q_probs)

        # Compute importance-sampling weights and log action probabilities
        weights = self.buffer.importance_sampling(samples['priorities'])  # shape -> [batch_size]
        log_action_probs = torch.log(self.local_network(states))
        action_probs = log_action_probs[range(self.batch_size), samples['actions']]

        # Compute and minimize loss
        loss = -torch.sum(double_q * action_probs.cpu(), dim=1).to(self.device)  # Cross-entropy
        loss = (weights * loss).mean()

        # Add intrinsic loss
        if self.im_method is not None:
            loss, im_loss = self.im_method.module.compute_loss(im_exp, loss)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.local_network.parameters(), self.params.clip_grad)
        self.optimizer.step()

        # Update target network
        self.__update_target_network()

        # Log actions
        self.log_data(
            actions=Counter(samples['actions'].detach().cpu().tolist())
        )

        # Update buffer priorities
        self.buffer.update_priorities(samples['priority_indices'], loss.detach())
        avg_return = returns.mean().item()
        return avg_return, loss.item(), im_loss

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

    def compute_double_q(self, states: torch.Tensor, samples: dict, returns: torch.Tensor, double_q_probs: torch.Tensor) -> torch.Tensor:
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
        double_q = states.new_zeros(self.batch_size, self.params.n_atoms)  # 1D array
        offset = torch.linspace(0, ((self.batch_size - 1) * self.params.n_atoms), self.batch_size)
        offset = offset.unsqueeze(1).expand(self.batch_size, self.params.n_atoms).to(samples['actions'])

        # Add flattened bounded offsets to Double-Q probabilities
        lb, ub = (lower_bound + offset).flatten(), (upper_bound + offset).flatten()
        double_lb = (double_q_probs * (upper_bound.float() - projection)).flatten()
        double_ub = (double_q_probs * (projection - lower_bound.float())).flatten()
        double_q.flatten().index_add_(dim=0, index=lb, source=double_lb)  # Add lower bounds
        double_q.flatten().index_add_(dim=0, index=ub, source=double_ub)  # Add upper bounds
        return double_q

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
        # Output info to console
        buffer_idx, buffer_let = number_to_num_letter(self.buffer.capacity)
        timesteps_idx, timesteps_let = number_to_num_letter(self.params.max_timesteps)
        self._initial_output(num_episodes, f'Buffer size: {int(buffer_idx)}{buffer_let.lower()}, '
                                           f'batch size: {self.buffer.batch_size}, '
                                           f'max timesteps: {int(timesteps_idx)}{timesteps_let.lower()}, '
                                           f'num network updates: {self.params.update_steps}, '
                                           f'replay period: {self.params.replay_period}, '
                                           f'intrinsic method: {self.im_type}.')

        # Time training
        with timer('Total time taken:'):
            self.save_batch_time = datetime.now()  # print_every start time
            # Iterate over episodes
            for i_episode in range(1+custom_ep_start, num_episodes+1+custom_ep_start):
                # Initialize priority weight increase per episode
                self.priority_weight_increase = min(i_episode / num_episodes, 1)
                state = self.env.reset()  # Initialize state
                score = 0.
                avg_returns, train_losses = [], []
                im_losses = []

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
                            avg_return, train_loss, im_loss = self.learn()

                            # Add items to lists
                            avg_returns.append(avg_return)
                            train_losses.append(train_loss)

                            if im_loss is not None:
                                im_losses.append(im_loss)

                    # Update state
                    state = next_state

                    # Check if finished
                    if done:
                        break

                # Sample new noise every replay period
                if i_episode % self.params.replay_period == 0:
                    self.local_network.sample_noise(self.device)

                # Add episode data to logger
                self.log_data(
                    ep_scores=score,
                    avg_returns=avg_returns[-1],
                    train_losses=train_losses[-1]
                )

                # Add intrinsic loss to logger if available
                if self.im_method is not None:
                    im_loss = im_losses[-1] if self.im_type == ValidIMMethods.EMPOWERMENT.value else im_losses[-1].item()
                    self.log_data(intrinsic_losses=im_loss)

                # Display output and save model
                model_name = f'rainbow-{self.im_type[:3]}' if self.im_type is not None else 'rainbow'
                buffer_idx, buffer_letter = number_to_num_letter(self.buffer.capacity)
                self.__output_progress(num_episodes+custom_ep_start, i_episode, print_every, custom_ep_start)
                extra_data = {
                    'local_network': self.local_network.state_dict(),
                    'target_network': self.target_network.state_dict(),
                    'network_type': self.local_network.__class__.__name__,
                    'optimizer': self.optimizer,
                    'buffer_params': self.buffer_params,
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
                                           filename=f'{model_name}_batch{self.buffer.batch_size}_'
                                                    f'buffer{int(buffer_idx)}{buffer_letter.lower()}',
                                           extra_data=extra_data,
                                           custom_ep_count=custom_ep_start)
            print(f"Training complete. Access metrics from 'logger' attribute.", end=' ')

    @staticmethod
    def __target_hard_update_loop(target_params, local_params) -> None:
        """Helper function for hard updating target networks."""
        for target, local in zip(target_params, local_params):
            target.data.copy_(local.data)

    def __update_target_network(self) -> None:
        """
        Performs a soft update of the target networks parameters.
        Formula: θ_target = Τ * θ_local + (1 - Τ) * θ_target
        """
        for target, local in zip(self.target_network.parameters(), self.local_network.parameters()):
            target.data.copy_(self.params.tau * local.data + (1.0 - self.params.tau) * target.data)

        if self.im_type == ValidIMMethods.EMPOWERMENT.value:
            emp_model = self.im_method.model
            self.__target_hard_update_loop(emp_model.source_target.parameters(), emp_model.source_net.parameters())
            self.__target_hard_update_loop(emp_model.forward_target.parameters(), emp_model.forward_net.parameters())

    def __output_progress(self, num_episodes: int, i_episode: int, print_every: int, custom_ep_start: int = 0) -> None:
        """Provides a progress update on the model's training to the console."""
        first_episode = i_episode == 1+custom_ep_start
        last_episode = i_episode == num_episodes

        if first_episode or last_episode or i_episode % print_every == 0:
            ep_idx, ep_letter = number_to_num_letter(i_episode)  # 1000 -> 1K
            ep_total_idx, ep_total_letter = number_to_num_letter(num_episodes)
            time_taken = (datetime.now() - self.save_batch_time)

            print(f'({ep_idx:.1f}{ep_letter}/{int(ep_total_idx)}{ep_total_letter})  ', end='')
            print(f'Episode Score: {int(self.logger.ep_scores[i_episode-1])},  '
                  f'Train Loss: {self.logger.train_losses[i_episode-1]:.5f},  ', end='')

            if self.im_method is not None:
                intrinsic_losses = self.logger.intrinsic_losses[i_episode - 1]
                if isinstance(intrinsic_losses, list):
                    print(f'Source Loss: {intrinsic_losses[0]:.5f},  Forward Loss: {intrinsic_losses[1]:.5f},  ', end='')
                else:
                    print(f'{self.im_type.title()} Loss: {intrinsic_losses:.5f},  ', end='')

            print(timer_string(time_taken, 'Time taken:'))
            self.save_batch_time = datetime.now()  # Reset
