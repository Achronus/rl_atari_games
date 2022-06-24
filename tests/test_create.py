import pytest

from core.create import create_model, SetModels, YamlParameters, CheckParamsValid
from core.exceptions import MissingVariableError

import torch


@pytest.fixture
def yaml_params() -> YamlParameters:
    return YamlParameters('parameters')


@pytest.fixture
def device() -> str:
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def test_create_model_valid(device) -> None:
    try:
        model = create_model('dqn', device)
        assert True
    except (AssertionError, MissingVariableError):
        assert False


def test_create_model_invalid_model_type(device) -> None:
    try:
        model = create_model('test', device)
        assert False
    except ValueError:
        assert True


def test_set_models_valid(yaml_params, device) -> None:
    try:
        set_model = SetModels(yaml_params, device)
        assert True
    except MissingVariableError:
        assert False


def test_yaml_parameters_invalid() -> None:
    try:
        yaml_params = YamlParameters('no_file')
        assert False
    except AssertionError:
        assert True


def test_check_params_valid(yaml_params) -> None:
    try:
        CheckParamsValid('rainbow', yaml_params)
        assert True
    except (AssertionError, MissingVariableError):
        assert False
