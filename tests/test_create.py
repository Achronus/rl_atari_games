import pytest

from core.create import create_model, SetModels, YamlParameters, CheckParamsValid
from core.exceptions import MissingVariableError
from core.parameters import EnvParameters


@pytest.fixture
def yaml_params() -> YamlParameters:
    return YamlParameters('parameters')


def test_create_model_valid() -> None:
    try:
        model = create_model('dqn')
        assert True
    except (AssertionError, MissingVariableError):
        assert False


def test_create_model_invalid_model_type() -> None:
    try:
        model = create_model('test')
        assert False
    except ValueError:
        assert True


def test_set_models_valid(yaml_params) -> None:
    try:
        set_model = SetModels(yaml_params)
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
