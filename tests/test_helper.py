from utils.helper import *

import torch


def test_set_device_custom_valid_cpu() -> None:
    device, _ = set_devices('cpu')
    assert True if device == 'cpu' else False


def test_number_to_num_letter_valid() -> None:
    num, letter = number_to_num_letter(1000)
    assert num == 1 and letter == 'K'


def test_to_tensor_list_valid() -> None:
    tensor = to_tensor([5, 6, 7])
    assert all(torch.eq(tensor, torch.Tensor([5, 6, 7])))


def test_to_tensor_numpy_valid() -> None:
    tensor = to_tensor(np.array([5, 6, 7]))
    assert all(torch.eq(tensor, torch.Tensor([5, 6, 7])))


def test_save_model_valid() -> None:
    try:
        save_model('test', {})
        os.remove('saved_models/test.pt')
        assert True
    except (ValueError, FileNotFoundError):
        assert False
