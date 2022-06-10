from utils.logger import DQNLogger, PPOLogger


def test_dqn_logger_add_valid() -> None:
    try:
        logger = DQNLogger()
        logger.add(actions=1)
        assert True
    except ValueError:
        assert False


def test_ppo_logger_add_valid() -> None:
    try:
        logger = PPOLogger()
        logger.add(rewards=[1, 1, 1])
        assert True
    except ValueError:
        assert False
