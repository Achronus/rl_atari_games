from utils.logger import DQNLogger, PPOLogger, RDQNLogger


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


def test_rdqn_logger_add_valid() -> None:
    try:
        logger = RDQNLogger()
        logger.add(returns=[1, 1, 1])
        assert True
    except ValueError:
        assert False
