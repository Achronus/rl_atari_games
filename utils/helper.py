"""
Helper functions that are usable across the whole application.
"""
from contextlib import contextmanager
import math
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Union

from utils.init_devices import CUDADevices

import torch


def set_device(device: str = None, threshold: float = 2e9) -> str:
    """
    Gets a string defining CUDA or CPU based on GPU availability.

    :param device (str) - an optional parameter that defines a CUDA device to use
    :param threshold (float) - an acceptance threshold for devices with higher available memory (default: ~2GB)
    """
    devices = CUDADevices(threshold)
    devices.set_device(device)
    return devices.device


def to_tensor(x: Union[list, np.array, torch.Tensor]) -> torch.Tensor:
    """Converts a list or numpy array to a torch tensor."""
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asarray(x))


def normalize(data: Union[list, np.array, torch.Tensor]) -> Union[np.array, torch.Tensor]:
    """Normalize a given list, array or tensor of data and return it."""
    if not isinstance(data, torch.Tensor):
        data = np.asarray(data)
    return (1.0 / 255) * data


def number_to_num_letter(num: int) -> tuple:
    """Converts a provided integer into a human-readable format. E.g, 1000 -> 1k. Returns the number and letter."""
    letters = ['', 'K', 'M', 'G', 'T', 'P']
    condition = 0 if num == 0 else math.log10(abs(num)) / 3
    idx = max(0, min(len(letters)-1, int(math.floor(condition))))
    num /= 10 ** (3 * idx)
    return num, letters[idx]


def save_model(filename: str, param_dict: dict) -> None:
    """Saves a model's state dict and config object to the saved_models folder."""
    os.makedirs('saved_models', exist_ok=True)
    torch.save(param_dict, f'saved_models/{filename}.pt')


@contextmanager
def timer(message: str = '') -> None:
    """A context manager that calculates the time a block of code takes to run."""
    start = datetime.now()  # Start timer
    yield
    end = datetime.now()  # End timer
    time_elapsed = end - start
    return timer_string(time_elapsed, message)


def timer_string(time_elapsed: timedelta, message: str = '') -> str:
    """
    Helper function for outputting a timer string.
    Example: 01:02:13 -> '1 hrs, 2 mins, 13 secs'

    :param time_elapsed (datetime.timedelta) - a timedelta containing a datatime time
    :param message (str) - an optional message prepended to the front of the returned string
    """
    split_time = str(time_elapsed).split(':')
    hrs, mins = [int(item) for item in split_time[:2]]
    secs = float(split_time[-1])

    # Set time string
    time_string = ''
    if hrs > 0:
        time_string += f'{hrs} hrs, '
    if mins > 0:
        time_string += f'{mins} mins, '
    time_string += f'{secs:.2f} secs'

    return f'{message} {time_string}.'


def dict_search(data_dict: dict) -> tuple:
    """Iterates over a dictionary recursively. Returns each key-value pair."""
    for key, value in data_dict.items():
        if isinstance(value, dict):
            for item in dict_search(value):
                yield item
        else:
            yield key, value
