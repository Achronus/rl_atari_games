import os
from dotenv import load_dotenv

from core.env_details import EnvDetails
from core.parameters import ModelParameters, DQNParameters
from agents.dqn import DQN
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
    # Set classes
    env_details = EnvDetails(
        gym_name=os.getenv('ENV_1'),
        img_size=int(os.getenv('IMG_SIZE')),
        stack_size=int(os.getenv('STACK_SIZE'))
    )

    network = CNNModel(input_shape=env_details.input_shape,
                       n_actions=env_details.n_actions, seed=SEED)

    model_params = ModelParameters(
        network=network,
        optimizer=optim.Adam(network.parameters(), lr=LEARNING_RATE,
                             eps=EPSILON),
        loss_metric=nn.MSELoss()
    )

    dqn_params = DQNParameters(
        gamma=float(os.getenv('GAMMA')),
        tau=float(os.getenv('TAU')),
        buffer_size=int(float(os.getenv('BUFFER_SIZE'))),
        batch_size=int(os.getenv('BATCH_SIZE')),
        update_steps=int(os.getenv('UPDATE_STEPS')),
        target_network=network,
        eps_start=float(os.getenv('EPS_START')),
        eps_end=float(os.getenv('EPS_END')),
        eps_decay=float(os.getenv('EPS_DECAY')),
        max_timesteps=int(os.getenv('MAX_TIMESTEPS'))
    )

    # Create DQN instance
    dqn = DQN(env_details, model_params, dqn_params, SEED)

    # Train model
    dqn.train(num_episodes=NUM_EPISODES, print_every=100)


if __name__ == '__main__':
    main()

