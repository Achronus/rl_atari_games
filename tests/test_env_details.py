from core.env_details import EnvDetails
from core.parameters import EnvParameters


def test_env_parameters_invalid_env_name() -> None:
    assert EnvParameters(100)


def test_set_envs_valid_img_size() -> None:
    env_details = EnvDetails(EnvParameters('ALE/SpaceInvaders-v5', 200, 4))
    assert env_details.input_shape == (4, 200, 200)


def test_set_envs_valid_stack() -> None:
    env_details = EnvDetails(EnvParameters('ALE/SpaceInvaders-v5', 128, 6))
    assert env_details.input_shape == (6, 128, 128)


def test_set_envs_valid() -> None:
    try:
        env_details = EnvDetails(EnvParameters('ALE/SpaceInvaders-v5', 128, 6))
        assert True
    except (ValueError, TypeError):
        assert False
