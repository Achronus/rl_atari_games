import os
import random
import numpy as np

import torch
from dotenv import load_dotenv

from agents.ppo import PPO
from core.parameters import (
    EnvParameters,
    DQNParameters,
    PPOParameters,
    ModelParameters
)
from core.env_details import EnvDetails
from agents.dqn import DQN
from models.actor_critic import ActorCritic
from models.cnn import CNNModel

import torch.optim as optim
import torch.nn as nn

load_dotenv()  # Create access to .env file

SEED = int(os.getenv('SEED'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
EPSILON = float(os.getenv('EPSILON'))
NUM_EPISODES = int(os.getenv('NUM_EPISODES'))
SAVE_EVERY = int(os.getenv('SAVE_EVERY'))
CAPTURE_VIDEO = True if os.getenv('CAPTURE_VIDEO') == 'True' else False

# Seeding
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def main() -> None:
    """Run main functionality of the application."""
    env_params = EnvParameters(
        env_name=os.getenv('ENV_1'),
        img_size=int(os.getenv('IMG_SIZE')),
        stack_size=int(os.getenv('STACK_SIZE')),
        capture_video=CAPTURE_VIDEO,
        record_every=SAVE_EVERY,
        seed=SEED
    )

    # Set classes instances
    env_details = EnvDetails(env_params)

    network = CNNModel(input_shape=env_details.input_shape, n_actions=env_details.n_actions)

    dqn_model_params = ModelParameters(
        network=network,
        optimizer=optim.Adam(network.parameters(), lr=LEARNING_RATE, eps=EPSILON),
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
        max_timesteps=int(os.getenv('MAX_TIMESTEPS'))
    )

    actor_critic = ActorCritic(input_shape=env_details.input_shape, n_actions=env_details.n_actions)

    ppo_model_params = ModelParameters(
        network=actor_critic,
        optimizer=optim.Adam(actor_critic.parameters(), lr=LEARNING_RATE, eps=EPSILON),
        loss_metric=nn.MSELoss()
    )

    ppo_params = PPOParameters(
        gamma=float(os.getenv('GAMMA')),
        update_steps=int(os.getenv('UPDATE_STEPS')),
        clip_grad=float(os.getenv('CLIP_GRAD')),
        rollout_size=int(os.getenv('ROLLOUT_SIZE')),
        num_agents=int(os.getenv('NUM_AGENTS')),
        num_mini_batches=int(os.getenv('NUM_MINI_BATCHES')),
        entropy_coef=float(os.getenv('ENTROPY_COEF')),
        value_loss_coef=float(os.getenv('VALUE_LOSS_COEF')),
        max_grad_norm=float(os.getenv('MAX_GRAD_NORM'))
    )

    # Create agent instances
    dqn = DQN(env_details, dqn_model_params, dqn_params, SEED)
    ppo = PPO(env_details, ppo_model_params, ppo_params, SEED)

    # Train models
    dqn.train(num_episodes=NUM_EPISODES, print_every=100, save_count=SAVE_EVERY)
    torch.cuda.empty_cache()  # Reset cache before training next algorithm
    ppo.train(num_episodes=NUM_EPISODES, print_every=100, save_count=SAVE_EVERY)


if __name__ == '__main__':
    main()
