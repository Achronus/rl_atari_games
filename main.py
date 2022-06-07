import os
from dotenv import load_dotenv

from agents.ppo import PPO
from core.env_details import EnvDetails
from core.parameters import DQNModelParameters, DQNParameters, PPOModelParameters, PPOParameters
from agents.dqn import DQN
from models.actor_critic import Actor, Critic
from models.cnn import CNNModel

import torch.optim as optim
import torch.nn as nn

load_dotenv()  # Create access to .env file

SEED = int(os.getenv('SEED'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
EPSILON = float(os.getenv('EPSILON'))
NUM_EPISODES = int(os.getenv('NUM_EPISODES'))


def main() -> None:
    """Run main functionality of the application."""
    # Set classes instances
    env_details = EnvDetails(
        gym_name=os.getenv('ENV_1'),
        img_size=int(os.getenv('IMG_SIZE')),
        stack_size=int(os.getenv('STACK_SIZE'))
    )

    network = CNNModel(input_shape=env_details.input_shape, n_actions=env_details.n_actions, seed=SEED)

    dqn_model_params = DQNModelParameters(
        network=network,
        optimizer=optim.Adam(network.parameters(), lr=LEARNING_RATE, eps=EPSILON),
        loss_metric=nn.MSELoss()
    )

    actor = Actor(input_shape=env_details.input_shape, n_actions=env_details.n_actions, seed=SEED)
    critic = Critic(input_shape=env_details.input_shape, n_actions=env_details.n_actions, seed=SEED)

    ppo_model_params = PPOModelParameters(
        actor=actor,
        critic=critic,
        actor_optimizer=optim.Adam(network.parameters(), lr=LEARNING_RATE, eps=EPSILON),
        critic_optimizer=optim.Adam(critic.parameters(), lr=LEARNING_RATE, eps=EPSILON),
        loss_metric=nn.MSELoss()
    )

    dqn_params = DQNParameters(
        gamma=float(os.getenv('GAMMA')),
        tau=float(os.getenv('TAU')),
        buffer_size=int(float(os.getenv('BUFFER_SIZE'))),
        batch_size=int(os.getenv('BATCH_SIZE')),
        update_steps=int(os.getenv('UPDATE_STEPS')),
        eps_start=float(os.getenv('EPS_START')),
        eps_end=float(os.getenv('EPS_END')),
        eps_decay=float(os.getenv('EPS_DECAY')),
        max_timesteps=int(os.getenv('DQN_MAX_TIMESTEPS'))
    )

    ppo_params = PPOParameters(
        gamma=float(os.getenv('GAMMA')),
        update_steps=int(os.getenv('UPDATE_STEPS')),
        clip_grad=float(os.getenv('CLIP_GRAD')),
        rollout_size=int(os.getenv('ROLLOUT_SIZE')),
        max_timesteps=int(os.getenv('PPO_MAX_TIMESTEPS'))
    )

    # Create agent instances
    dqn = DQN(env_details, dqn_model_params, dqn_params, SEED)
    ppo = PPO(env_details, ppo_model_params, ppo_params, SEED)

    # Train models
    # dqn.train(num_episodes=NUM_EPISODES, print_every=100)
    ppo.train(num_episodes=5, print_every=1)


if __name__ == '__main__':
    main()

