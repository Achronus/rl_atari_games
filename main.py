import os
from dotenv import load_dotenv

from core.create import create_model

import torch

load_dotenv()  # Create access to .env file

NUM_EPISODES = int(os.getenv('NUM_EPISODES'))
SAVE_EVERY = int(os.getenv('SAVE_EVERY'))


def main():
    """Run main functionality of the application."""
    # Create agent instances
    dqn = create_model('dqn')
    ppo = create_model('ppo')

    # Train models
    dqn.train(num_episodes=NUM_EPISODES, print_every=100, save_count=SAVE_EVERY)
    torch.cuda.empty_cache()  # Reset cache before training next algorithm
    ppo.train(num_episodes=NUM_EPISODES, print_every=100, save_count=SAVE_EVERY)


if __name__ == '__main__':
    main()
