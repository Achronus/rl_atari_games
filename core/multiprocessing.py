"""
Contains the functionality for multi-GPU processing.
"""
import os


from core.enums import ValidModels, ValidIMMethods
from intrinsic.empower_models import RainbowNetwork
from models.dueling import CategoricalNoisyDueling

import torch
from torch import optim
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def multiprocess_cleanup() -> None:
    """Tidies up finished process groups."""
    dist.destroy_process_group()


def init_distributed_backend(rank: int, world_size: int) -> None:
    """Initializes the distributed backend for synchronizing GPUs when using multiple devices."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank
    )


class MultiProcessingHelper:
    """A helper class for creating model and optimizers within the multi_train function."""
    @staticmethod
    def create_rainbow_model(im_name: str, input_shape: tuple, n_actions: int) -> nn.Module:
        """Helper function for creating a rainbow model based on the given parameters."""
        model = RainbowNetwork if im_name == ValidIMMethods.EMPOWERMENT.value else CategoricalNoisyDueling
        return model(input_shape, n_actions)


def create_model(model_type: str, im_name: str, input_shape: tuple, n_actions: int) -> nn.Module:
    """Creates a model based on the given parameters."""
    for item in ValidModels:
        if model_type == item.value:
            func_name = f'create_{model_type}_model'
            func = getattr(MultiProcessingHelper, func_name)
            return func(im_name, input_shape, n_actions)


def multi_train(rank: int, world_size: int, model_name: str) -> None:
    """
    Performs a subset of model training using multiple devices. Reduces write overheads for Distributed Data
    Parallel (DDP) by saving a checkpoint on one device and loads it's into all devices. See official PyTorch docs
    for more details: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#save-and-load-checkpoints.

    :param rank (int) - id of current device
    :param world_size (int) - total number of devices
    :param model_name (str) - the name of the model used
    """
    checkpoint_path = "saved_models/model.checkpoint"
    network_args_path = "saved_models/network_args.checkpoint"
    not_ppo = model_name != ValidModels.PPO.value
    map_location = {'cuda:0': f'cuda:{rank}'}
    network_args = torch.load(network_args_path, map_location=map_location)

    # Initialize models for multiprocessing
    init_distributed_backend(rank, world_size)
    if not_ppo:
        local_network = DDP(create_model('rainbow', network_args['im_name'], network_args['input_shape'],
                                         network_args['n_actions']).to(rank))
        target_network = DDP(create_model('rainbow', network_args['im_name'], network_args['input_shape'],
                                          network_args['n_actions']).to(rank))
    else:
        network = DDP(create_model('ppo', network_args['im_name'], network_args['input_shape'],
                                   network_args['n_actions']).to(rank))

    # Create temporary checkpoint when saving
    if rank == 0:
        # DQN
        if not_ppo:
            torch.save({
                'local_network': local_network.state_dict(),
                'target_network': target_network.state_dict()
            }, checkpoint_path)
        # PPO
        else:
            torch.save(network.state_dict(), checkpoint_path)

    # Wait for processes to finish
    dist.barrier()

    # Assign approach checkpoint to relevant device
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # DQN
    if not_ppo:
        local_network.load_state_dict(checkpoint.get('local_network'), strict=False)
        target_network.load_state_dict(checkpoint.get('target_network'), strict=False)
    # PPO
    else:
        network.load_state_dict(checkpoint, strict=False)

    optimizer = optim.Adam(local_network.parameters(), lr=0.001, eps=0.001)

    optimizer.zero_grad()
    network_args['loss'].backward()
    nn.utils.clip_grad_norm_(local_network.parameters(), network_args['clip_grad'])
    optimizer.step()

    multiprocess_cleanup()
