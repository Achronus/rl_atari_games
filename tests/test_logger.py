from utils.logger import DQNLogger


def test_logger_add_valid() -> None:
    try:
        logger = DQNLogger()
        logger.add(actions=1)
        assert True
    except ValueError:
        assert False
