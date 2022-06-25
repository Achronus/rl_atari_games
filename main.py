from core.create import create_model, get_utility_params
from utils.helper import set_device

# Get utility parameters from yaml file
util_params = get_utility_params()

# Set hyperparameters
NUM_EPISODES = util_params['num_episodes']
SAVE_EVERY = util_params['save_every']
env2 = util_params['env_2']
env3 = util_params['env_3']


def main():
    """Run main functionality of the application."""
    device = set_device()

    # Train on multiple environments
    for env in ['primary', env2, env3]:
        # Create agent instances
        dqn = create_model('rainbow', env=env, device=device)
        ppo = create_model('ppo', env=env, device=device)

        # Train model
        dqn.train(num_episodes=NUM_EPISODES, print_every=1000, save_count=SAVE_EVERY)
        ppo.train(num_episodes=NUM_EPISODES, print_every=1000, save_count=SAVE_EVERY)


if __name__ == '__main__':
    main()
