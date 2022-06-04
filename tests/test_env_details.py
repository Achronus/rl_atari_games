from core.env_details import EnvDetails


def test_set_envs_invalid_env_names() -> None:
    try:
        env_details = EnvDetails(gym_name=100)
        assert False
    except AssertionError:
        assert True


def test_set_envs_invalid_img_size() -> None:
    try:
        env_details = EnvDetails(gym_name='ALE/SpaceInvaders-v5', img_size=20.5)
        assert False
    except AssertionError:
        assert True


def test_set_envs_invalid_stack_size() -> None:
    try:
        env_details = EnvDetails(gym_name='ALE/SpaceInvaders-v5', stack_size=0.5)
        assert False
    except AssertionError:
        assert True


def test_set_envs_valid_img_size() -> None:
    env_details = EnvDetails(gym_name='ALE/SpaceInvaders-v5', img_size=200)
    assert env_details.input_shape == (4, 200, 200)


def test_set_envs_valid_stack() -> None:
    env_details = EnvDetails(gym_name='ALE/SpaceInvaders-v5', img_size=128, stack_size=6)
    assert env_details.input_shape == (6, 128, 128)


def test_set_envs_valid() -> None:
    try:
        env_details = EnvDetails(gym_name='ALE/SpaceInvaders-v5')
        assert False
    except AssertionError:
        assert True
