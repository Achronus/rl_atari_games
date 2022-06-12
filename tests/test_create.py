from core.create import create_model, SetModels
from core.exceptions import MissingVariableError


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


def test_set_models_valid() -> None:
    try:
        set_model = SetModels()
        assert True
    except MissingVariableError:
        assert False
