import warnings
import argparse

from core.create import create_model, get_utility_params
from utils.helper import set_device

# Get utility parameters from yaml file
util_params = get_utility_params()

# Set hyperparameters
NUM_EPISODES = util_params['num_episodes']
SAVE_EVERY = util_params['save_every']
env2 = util_params['env_2']
env3 = util_params['env_3']

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    """Run main functionality of the application."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_name', help="An optional custom CUDA device name.", type=str, nargs='?', const=None)
    args = parser.parse_args()

    if args.device_name is not None:
        device = set_device(args.device_name)
    else:
        device = set_device()

    # Train on multiple environments
    # for env in ['primary', env2, env3]:
    #     model = create_model('dqn', env=env, device=device)
    #     model.train(num_episodes=NUM_EPISODES, print_every=1000, save_count=SAVE_EVERY)

    # Train on single environment
    model = create_model('dqn', env='primary', device=device)
    model.train(num_episodes=NUM_EPISODES, print_every=1000, save_count=SAVE_EVERY)


if __name__ == '__main__':
    main()
