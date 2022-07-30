import os
from typing import Union

from core.create import create_model, get_utility_params
from core.multiprocessing import run
from utils.helper import set_devices

import torch.multiprocessing as mp

# Get utility parameters from yaml file
util_params = get_utility_params()

# Set hyperparameters
NUM_EPISODES = util_params['num_episodes']
SAVE_EVERY = util_params['save_every']
env2 = util_params['env_2']
env3 = util_params['env_3']


def main():
    """Run main functionality of the application."""
    device, multi_devices = set_devices()

    for env in ['primary', env2, env3]:
        for im_method in [None, 'curiosity', 'empowerment']:
            # Create agent instances
            dqn = create_model('rainbow', env=env, devices=(device, multi_devices), im_type=im_method)
            ppo = create_model('ppo', env=env, devices=(device, multi_devices), im_type=im_method)
            train_args = {'num_episodes': NUM_EPISODES, 'print_every': 1000, 'save_count': SAVE_EVERY}

            # Initialize distribution if multi-devices
            if multi_devices is not None:
                indices = ','.join([item[-1] for item in multi_devices])
                os.environ['CUDA_VISIBLE_DEVICES'] = indices
                world_size = len(multi_devices)

                for model in [dqn, ppo]:
                    mp.spawn(
                        run,
                        args=(world_size, model.train, train_args),
                        nprocs=world_size,
                        join=True
                    )
            else:
                for model in [dqn, ppo]:
                    model.train(**train_args)


if __name__ == '__main__':
    main()
