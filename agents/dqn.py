import random
from collections import namedtuple

from core.buffer import ReplayBuffer
from core.parameters import ModelParameters, DQNParameters
from core.env_details import EnvDetails
from utils.helper import *
from utils.logger import DQNLogger

Experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])


class DQN:
    """
    A basic Deep Q-Network that uses an experience replay buffer and fixed Q-targets.

    Parameters:
        env_details (EnvDetails) - a class containing parameters for the environment
        model_params (ModelParameters) - a data class containing model specific parameters
        dqn_params (DQNParameters) - a data class containing DQN specific parameters
        seed (int) - an integer for recreating results
    """
    def __init__(self, env_details: EnvDetails, model_params: ModelParameters,
                 dqn_params: DQNParameters, seed: int) -> None:
        self.env_details = env_details
        self.model_params = model_params
        self.params = dqn_params
        self.logger = DQNLogger()
        self.seed = seed

        self.env = env_details.env
        self.state_size = env_details.stack_size
        self.action_size = env_details.n_actions
        random.seed(seed)

        self.device: str = set_device()

        self.local_network = model_params.network.to(self.device)
        self.target_network = dqn_params.target_network.to(self.device)  # Fixed target network
        self.optimizer = model_params.optimizer
        self.loss = model_params.loss_metric

        self.memory = ReplayBuffer(env_details, dqn_params.buffer_size, dqn_params.batch_size, self.device, seed)
        self.timestep = 0

    def step(self, experience: Experience) -> None:
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
        state = state.unsqueeze(0).to(self.device)

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
        self.logger.add(**dict(
                q_targets_next=q_targets_next,
                q_targets=q_targets,
                q_preds=q_preds,
                train_losses=loss
            )
        )

        # Update target network
        self.__soft_update()

    def __soft_update(self) -> None:
        """
        Performs a soft update of the target networks parameters.
        Formula: θ_target = Τ * θ_local + (1 - Τ) * θ_target
        """
        for target, local in zip(self.target_network.parameters(), self.local_network.parameters()):
            target.data.copy_(self.params.tau * local.data + (1.0 - self.params.tau) * target.data)

    def train(self, num_episodes: int, print_every: int = 100, save_count: int = 1000) -> None:
        """
        Train the agent.

        Parameters:
            num_episodes (int) - the number of iterations to train the agent on
            print_every (int) - the number of episodes before outputting information
            save_count (int) - the number of episodes before saving the model
        """
        episode_scores = []
        eps = self.params.eps_start

        # Output info to console
        ep_total_idx, ep_total_letter = number_to_num_letter(num_episodes)
        buffer_idx, buffer_letter = number_to_num_letter(self.memory.buffer_size)
        timesteps_idx, timesteps_letter = number_to_num_letter(self.params.max_timesteps)
        print(f'Training agent on {int(ep_total_idx)}{ep_total_letter} '
              f'episodes with {int(timesteps_idx)}{timesteps_letter.lower()} timesteps.')
        print(f'Buffer size: {int(buffer_idx)}{buffer_letter.lower()}, batch size: {self.memory.batch_size}.')

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
                self.step(Experience(state.cpu(), action, reward, normalize(next_state), done))

                # Update state and score
                state = next_state
                score += reward

                # Log actions and environment info
                self.logger.add(actions=action, env_info=info)

                # Check if finished
                if done:
                    break

            # Log metrics
            self.logger.add(**dict(ep_scores=score, epsilons=eps))

            # Update values
            episode_scores.append(score)
            eps = max(self.params.eps_end, self.params.eps_decay * eps)  # Decrease epsilon

            # Display output and save model
            self.__output_progress(num_episodes, i_episode, print_every, score)
            self.__save_model_condition(i_episode, save_count)
        print(f"Training complete. Access metrics from 'logger' attribute.")

    def __save_model_condition(self, i_episode: int, save_count: int) -> None:
        """Saves the model when the current episode equals the save count."""
        if i_episode % save_count == 0:
            ep_idx, ep_letter = number_to_num_letter(i_episode)
            filename = f'dqn_batch{self.memory.batch_size}_ep{ep_idx}{ep_letter}'.lower()

            # Save model
            save_model(filename, dict(
                local_network=self.local_network.state_dict(),
                target_network=self.target_network.state_dict(),
                env_details=self.env_details,
                model_params=self.model_params,
                dqn_params=self.params,
                logger=self.logger,
                seed=self.seed
            ))
            print(f"Saved model at episode {i_episode} as: '{filename}'.")

    def __output_progress(self, num_episodes: int, i_episode: int, print_every: int, score: int) -> None:
        """Provides a progress update on the model's training to the console."""
        first_episode = i_episode == 1
        last_episode = i_episode == num_episodes+1

        if first_episode or last_episode or i_episode % print_every == 0:
            ep_idx, ep_letter = number_to_num_letter(i_episode)  # 1000 -> 1K
            ep_total_idx, ep_total_letter = number_to_num_letter(num_episodes)

            print(f'({int(ep_idx)}{ep_letter}/{int(ep_total_idx)}{ep_total_letter}) ', end='')
            print(f'Episode Score: {int(score)}, Train Loss: {self.logger.train_losses[i_episode]:.5f}')


def load_dqn_model(filename: str, device: str) -> DQN:
    """Load model's parameters from the given filename. Files must be stored within a saved_models folder."""
    assert os.path.exists('saved_models'), "'saved_models' folder does not exist! Have you created it?"
    assert os.path.exists(f'saved_models/{filename}.pt'), "'filename' does not exist in the 'saved_models' folder!"

    checkpoint = torch.load(f'saved_models/{filename}.pt', map_location=device)
    env_details = checkpoint.get('env_details')
    model_params = checkpoint.get('model_params')
    dqn_params = checkpoint.get('dqn_params')
    logger = checkpoint.get('logger')
    seed = checkpoint.get('seed')

    dqn = DQN(env_details, model_params, dqn_params, seed)
    dqn.local_network.load_state_dict(checkpoint.get('local_network'), strict=False)
    dqn.target_network.load_state_dict(checkpoint.get('target_network'), strict=False)
    dqn.logger = logger
    print(f"Loaded DQN model: '{filename}'.")
    return dqn
