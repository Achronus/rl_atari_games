import os

from core.create import create_model, get_utility_params
from utils.helper import set_devices

import torch
import torch.distributed as dist

# Get utility parameters from yaml file
util_params = get_utility_params()

# Set them as hyperparameters
NUM_EPISODES = util_params['num_episodes']
SAVE_EVERY = util_params['save_every']


def init_distributed_backend(multi_devices: list) -> None:
    """Initializes the distributed backend for synchronizing GPUs when using multiple devices."""
    indices = ','.join([item[-1] for item in multi_devices])
    os.environ['CUDA_VISIBLE_DEVICES'] = indices

    dist_url = "env://"
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
        backend="ncc1",
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

    # Localize device calls
    torch.cuda.set_device(local_rank)

    # Synchronize all threads before creating models
    dist.barrier()


def main():
    """Run main functionality of the application."""
    # Set GPU device(s)
    device, multi_devices = set_devices()

    # Initialize distribution if multi-devices
    if multi_devices is not None:
        init_distributed_backend(multi_devices)

    # Create agent instances
    dqn = create_model('rainbow', devices=(device, multi_devices))
    ppo = create_model('ppo', devices=(device, multi_devices))

    # Train models
    dqn.train(num_episodes=NUM_EPISODES, print_every=100, save_count=SAVE_EVERY)
    torch.cuda.empty_cache()  # Reset cache before training next algorithm

    ppo_num_episodes = ppo.params.rollout_size * ppo.params.num_agents * NUM_EPISODES
    ppo.train(num_episodes=ppo_num_episodes, print_every=100, save_count=SAVE_EVERY)


if __name__ == '__main__':
    main()
