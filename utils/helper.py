"""
Helper functions that are usable across the whole application.
"""
from contextlib import contextmanager
import math
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Union

import torch


def set_device() -> str:
    """Gets a string defining CUDA or CPU based on GPU availability."""
    if torch.cuda.is_available():
        print(f'CUDA available. Device set to GPU.')
        return 'cuda:0'

    print("CUDA unavailable. Device set to CPU.")
    return 'cpu'


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


def number_to_num_letter(num: int) -> tuple[int, str]:
    """Converts a provided integer into a human-readable format. E.g, 1000 -> 1k."""
    letters = ['', 'K', 'M', 'G', 'T', 'P']
    condition = 0 if num == 0 else math.log10(abs(num)) / 3
    idx = max(0, min(len(letters)-1, int(math.floor(condition))))
    num /= 10 ** (3 * idx)
    return num, letters[idx]


def save_model(filename: str, param_dict: dict) -> None:
    """Saves a model's state dict, config object and logger object to the saved_models folder."""
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

    Parameters:
        time_elapsed (datetime.timedelta) - a timedelta containing a datatime time
        message (str) - an optional message prepended to the front of the returned string
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
