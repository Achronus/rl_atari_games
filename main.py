from core.create import create_model, get_utility_params

import torch

# Get utility parameters from yaml file
util_params = get_utility_params()

# Set them as hyperparameters
NUM_EPISODES = util_params['num_episodes']
SAVE_EVERY = util_params['save_every']


def main():
    """Run main functionality of the application."""
    # Create agent instances
    dqn = create_model('rainbow')
    ppo = create_model('ppo')

    # Train models
    dqn.train(num_episodes=NUM_EPISODES, print_every=100, save_count=SAVE_EVERY)
    torch.cuda.empty_cache()  # Reset cache before training next algorithm

    ppo_num_episodes = ppo.params.rollout_size * ppo.params.num_agents * NUM_EPISODES
    ppo.train(num_episodes=ppo_num_episodes, print_every=100, save_count=SAVE_EVERY)


if __name__ == '__main__':
    main()
