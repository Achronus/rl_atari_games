import pytest

from core.env_details import EnvDetails
from core.parameters import EnvParameters
from intrinsic.controller import IMController
from intrinsic.model import CuriosityModel
from intrinsic.module import Curiosity
from intrinsic.parameters import CuriosityParameters, IMExperience

import torch

from utils.helper import to_tensor


@pytest.fixture
def env_details() -> EnvDetails:
    return EnvDetails(EnvParameters('ALE/SpaceInvaders-v5', img_size=128, stack_size=4, seed=1))


@pytest.fixture
def curiosity_params(env_details) -> CuriosityParameters:
    return CuriosityParameters(input_shape=env_details.input_shape,
                               n_actions=env_details.n_actions)

@pytest.fixture
def optim_params() -> dict:
    return {'lr': 0.001, 'eps': 0.003}

@pytest.fixture
def device() -> str:
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def curiosity_model(env_details, device) -> CuriosityModel:
    return CuriosityModel(env_details.input_shape, env_details.n_actions, device).to(device)


@pytest.fixture
def controller(curiosity_params, optim_params, device) -> IMController:
    return IMController('curiosity', curiosity_params, optim_params, device)


@pytest.fixture
def state(device) -> torch.Tensor:
    return torch.randn((32, 4, 128, 128), device=device).to(torch.float32)


@pytest.fixture
def actions(device) -> torch.Tensor:
    return torch.randint(low=1, high=6, size=(32,), device=device).to(torch.long)


@pytest.fixture
def experience(state, actions) -> IMExperience:
    return IMExperience(state, state, actions)


def test_imcontroller_invalid_type(curiosity_params, optim_params, device) -> None:
    with pytest.raises(ValueError):
        IMController('None', curiosity_params, optim_params, device)


def test_imcontroller_init_valid(controller) -> None:
    valid_model = type(controller.model) == CuriosityModel
    valid_module = type(controller.module) == Curiosity
    assert all([valid_model, valid_module])


def test_imcontroller_model_encode_valid(controller, state) -> None:
    encoded_state = controller.model.encode(state)
    assert encoded_state.shape == (32, 9216)


def test_imcontroller_model_predict_action_valid(controller, state) -> None:
    action_probs = controller.model.predict_action(state, state)
    assert action_probs.shape == (32, 6)


def test_imcontroller_model_forward_valid(controller, state, actions) -> None:
    encoded_state = controller.model.encode(state)
    out = controller.model.forward(encoded_state, actions)
    assert out.shape == (32, 9216)


def test_imcontroller_module_get_loss_valid(controller, experience) -> None:
    loss, im_loss = controller.module.compute_loss(experience, to_tensor(0.056))

    def check(item) -> bool:
        return torch.numel(item) == 1 and type(item.item()) == float

    valid_loss = check(loss)
    valid_im_loss = check(im_loss)
    assert all([valid_loss, valid_im_loss])


def test_imcontroller_module_compute_loss_valid(controller, experience) -> None:
    loss, im_loss = controller.module.compute_loss(experience, to_tensor(0.056))

    def check(item) -> bool:
        return torch.numel(item) == 1 and type(item.item()) == float

    valid_loss = check(loss)
    valid_im_loss = check(im_loss)
    assert all([valid_loss, valid_im_loss])


def test_imcontroller_module_compute_return_valid(controller, experience) -> None:
    reward = controller.module.compute_return(experience)
    assert torch.numel(reward) == 32
