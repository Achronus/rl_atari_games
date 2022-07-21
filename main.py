import os

from core.create import create_model, get_utility_params
from utils.helper import set_devices

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Get utility parameters from yaml file
util_params = get_utility_params()

# Set hyperparameters
NUM_EPISODES = util_params['num_episodes']
SAVE_EVERY = util_params['save_every']
env2 = util_params['env_2']
env3 = util_params['env_3']


def init_distributed_backend(multi_devices: list) -> int:
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

    # Return world size
    return world_size


def train_agents(device, multi_devices) -> None:
    """Runs the training of the agents."""
    # Train on multiple environments
    for env in ['primary', env2, env3]:
        # Create agent instances
        dqn = create_model('rainbow', env=env, devices=(device, multi_devices))
        ppo = create_model('ppo', env=env, devices=(device, multi_devices))
        ppo_num_episodes = ppo.params.rollout_size * ppo.params.num_envs * NUM_EPISODES

        # Train model
        dqn.train(num_episodes=NUM_EPISODES, print_every=1000, save_count=SAVE_EVERY)
        ppo.train(num_episodes=ppo_num_episodes, print_every=1000, save_count=SAVE_EVERY)


def main():
    """Run main functionality of the application."""
    device, multi_devices = set_devices()

    # Initialize distribution if multi-devices
    if multi_devices is not None:
        world_size = init_distributed_backend(multi_devices)
        mp.spawn(
            train_agents,
            args=(world_size, device, multi_devices),
            nprocs=world_size,
            join=True
        )
    else:
        train_agents(device, multi_devices)

    # TODO: update output progress to single and multi-GPU
    # TODO: update save model to multi-GPU


if __name__ == '__main__':
    main()
