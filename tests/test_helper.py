from utils.helper import *

import torch


def test_set_device_valid_gpu() -> None:
    device = set_device()
    device_available = torch.cuda.is_available()
    assert True if device_available and device == 'cuda:0' else False


def test_set_device_valid_cpu() -> None:
    device = set_device()
    device_available = torch.cuda.is_available()
    assert False if device_available and device == 'cpu' else True


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
    except ValueError or FileNotFoundError:
        assert False
