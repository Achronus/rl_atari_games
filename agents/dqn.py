from collections import namedtuple, Counter
from typing import Union
import numpy as np
import random
from datetime import datetime

from agents._agent import Agent
from core.buffer import ReplayBuffer
from core.enums import ValidIMMethods
from core.parameters import AgentParameters, ModelParameters
from core.env_details import EnvDetails
from intrinsic.parameters import IMExperience
from utils.helper import number_to_num_letter, normalize, to_tensor, timer, timer_string
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
        :param im_type (tuple[str, IMController]) - the type of intrinsic motivation to use with its controller
        """
        self.logger = DQNLogger()
        super().__init__(env_details, params, device, seed, self.logger, im_type)

        self.env = env_details.make_env('dqn')
        self.action_size = env_details.n_actions

        self.memory = ReplayBuffer(params.buffer_size, params.batch_size, self.device, seed)

        self.local_network = model_params.network.to(self.device)
        self.target_network = model_params.network.to(self.device)  # Fixed target network

        self.optimizer = model_params.optimizer
        self.loss = model_params.loss_metric

        self.timestep = 0
        self.i_episode = 0
        self.save_batch_time = datetime.now()  # init

    def step(self, experience: DQNExperience) -> Union[tuple, None]:
        """Perform a learning step, if there are enough samples in memory.
        Returns the training loss and Q-value predictions."""
        # Store experience in memory
        self.memory.add(experience)

        # Learning every few update_steps
        self.timestep = (self.timestep + 1) % self.params.update_steps

        # Perform learning
        if self.timestep == 0 and len(self.memory) > self.params.batch_size:
            experiences = self.memory.sample()
            train_loss, im_loss = self.learn(experiences)
            return train_loss, im_loss
        return None, None

    def act(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        """
        Returns an action for a given state based on an epsilon greedy policy.

        :param state (torch.Tensor) - current state
        :param epsilon (float) - current epsilon
        """
        state = state.unsqueeze(0)

        # Set to evaluation mode and get actions
        self.local_network.eval()
        with torch.no_grad():
            state = self.encode_state(state)
            action_values = self.local_network(state)

        # Set back to training mode
        self.local_network.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().numpy()).item()
        return random.choice(np.arange(self.action_size))

    def learn(self, experiences: namedtuple) -> tuple:
        """Updates the network parameters. Returns the training loss and average return."""
        states, actions, rewards, next_states, dones = experiences
        im_loss = None

        # Encode states if using empowerment
        states = self.encode_state(states)
        next_states = self.encode_state(next_states)

        # Get max predicted Q-value from target network
        q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q-targets for the current state
        q_targets = rewards + (self.params.gamma * q_targets_next * (1 - dones))

        # Add intrinsic reward
        if self.im_method is not None:
            im_exp = IMExperience(states, next_states, actions.squeeze())
            im_return = normalize(self.im_method.module.compute_return(im_exp))
            q_targets += im_return
            q_targets = q_targets.detach()

        # Get expected Q-values from local network
        q_preds = self.local_network(states).gather(1, actions)

        # Compute and minimize loss
        loss = self.loss(q_preds, q_targets.to(torch.float32))

        # Add intrinsic loss
        if self.im_method is not None:
            loss, im_loss = self.im_method.module.compute_loss(im_exp, loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.__update_target_network()

        return loss.item(), im_loss

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
        # Set initial epsilon
        eps = self.params.eps_start

        # Output info to console
        buffer_idx, buffer_let = number_to_num_letter(self.memory.buffer_size)
        timesteps_idx, timesteps_let = number_to_num_letter(self.params.max_timesteps)
        self._initial_output(num_episodes, f'Buffer size: {int(buffer_idx)}{buffer_let.lower()}, '
                                           f'batch size: {self.memory.batch_size}, '
                                           f'max timesteps: {int(timesteps_idx)}{timesteps_let.lower()}, '
                                           f'num network updates: {self.params.update_steps}, '
                                           f'intrinsic method: {self.im_type}')

        # Time training
        with timer('Total time taken:'):
            self.save_batch_time = datetime.now()  # print_every start time
            # Iterate over episodes
            for i_episode in range(1+custom_ep_start, num_episodes+1+custom_ep_start):
                # Initialize state and score
                score = 0
                state = self.env.reset()
                actions, train_losses, im_losses = [], [], []
                self.i_episode = i_episode

                # Iterate over timesteps
                for t in range(self.params.max_timesteps):
                    state = normalize(to_tensor(state)).to(self.device)
                    action = self.act(state, eps)  # Generate an action
                    next_state, reward, done, info = self.env.step(action)  # Take an action

                    # Perform learning
                    exp = DQNExperience(state.cpu(), action, reward, normalize(to_tensor(next_state)), done)
                    train_loss, im_loss = self.step(exp)

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

                    if im_loss is not None:
                        im_losses.append(im_loss)

                # Log episodic metrics
                self.log_data(
                    ep_scores=score,
                    actions=Counter(actions),
                    train_losses=train_losses[-1]
                )

                # Add intrinsic loss to logger if available
                if self.im_method is not None:
                    im_loss = im_losses[-1] if self.im_type == ValidIMMethods.EMPOWERMENT.value else im_losses[-1].item()
                    self.log_data(intrinsic_losses=im_loss)

                # Decrease epsilon
                eps = max(self.params.eps_end, self.params.eps_decay * eps)

                # Display output and save model
                model_name = f'dqn-{self.im_type[:3]}' if self.im_type is not None else 'dqn'
                self.__output_progress(num_episodes+custom_ep_start, i_episode, print_every, custom_ep_start)
                extra_data = {
                    'local_network': self.local_network.state_dict(),
                    'target_network': self.target_network.state_dict(),
                    'network_type': self.local_network.__class__.__name__,
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
                                           filename=f'{model_name}_batch{self.memory.batch_size}'
                                                    f'_buffer{int(buffer_idx)}{buffer_let.lower()}',
                                           extra_data=extra_data)
            print(f"Training complete. Access metrics from 'logger' attribute.", end=' ')

    def __output_progress(self, num_episodes: int, i_episode: int, print_every: int, custom_ep_start: int = 0) -> None:
        """Provides a progress update on the model's training to the console."""
        first_episode = i_episode == 1+custom_ep_start
        last_episode = i_episode == num_episodes+1

        if first_episode or last_episode or i_episode % print_every == 0:
            ep_idx, ep_letter = number_to_num_letter(i_episode)  # 1000 -> 1K
            ep_total_idx, ep_total_letter = number_to_num_letter(num_episodes)
            time_taken = (datetime.now() - self.save_batch_time)

            print(f'({ep_idx:.1f}{ep_letter}/{int(ep_total_idx)}{ep_total_letter}) ', end='')
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
