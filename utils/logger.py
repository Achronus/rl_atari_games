class Logger:
    """
    A base logger that stores information for each episode iteration.

    Parameters:
        keys (list[str]) - a list of names of information to store
    """
    def __init__(self, keys: list) -> None:
        self.keys = keys
        self.__set_defaults()

    def add(self, **kwargs) -> None:
        """Add episode items to respective lists in the logger."""
        for key, val in kwargs.items():
            if key not in self.keys:
                raise ValueError(f"Key does not exist! Must be one of: '{self.keys}'")

            # If actions, update counter values
            if key == 'actions' and len(getattr(self, key)) != 0:
                actions = getattr(self, key)
                actions[0] += val
            # Otherwise, add to list as normal
            else:
                getattr(self, key).append(val)

    def __set_defaults(self) -> None:
        """Creates empty list values for all keys."""
        for key in self.keys:
            setattr(self, key, [])

    def __repr__(self) -> str:
        return f"Available attributes: '{self.keys}'"


class DQNLogger(Logger):
    """A DQN logger that stores information for each episode iteration."""
    def __init__(self) -> None:
        self.keys = ['actions', 'train_losses', 'ep_scores', 'q_targets_next',
                     'q_targets', 'q_preds']
        super().__init__(self.keys)


class PPOLogger(Logger):
    """A PPO logger that stores information for each episode iteration."""
    def __init__(self) -> None:
        self.keys = ['actions', 'avg_rewards', 'avg_returns', 'policy_losses',
                     'value_losses', 'entropy_losses', 'total_losses',
                     'approx_kl']
        super().__init__(self.keys)


class RDQNLogger(Logger):
    """A Rainbow DQN logger that stores information for each episode iteration."""
    def __init__(self) -> None:
        self.keys = ['avg_returns', 'actions', 'train_losses', 'ep_scores']
        super().__init__(self.keys)
