class Logger:
    """
    A base logger that stores information for each episode iteration.

    Parameters:
        keys (list[str, ...]) - a list of names of information to store
    """
    def __init__(self, keys: list[str, ...]) -> None:
        self.keys = keys
        self.__set_defaults()

    def add(self, **kwargs) -> None:
        """Add episode items to respective lists in the logger."""
        for key, val in kwargs.items():
            if key not in self.keys:
                raise ValueError(f"Key does not exist! Must be one of: '{self.keys}'")

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
        self.keys = ['actions', 'env_info', 'train_losses', 'ep_scores', 'epsilons',
                     'q_targets_next', 'q_targets', 'q_preds']
        super().__init__(self.keys)


class PPOLogger(Logger):
    """A PPO logger that stores information for each episode iteration."""
    def __init__(self) -> None:
        self.keys = ['predictions', 'env_info', 'rewards', 'advantages', 'returns',
                     'ratios', 'log_ratios', 'policy_losses', 'value_losses',
                     'entropy_losses', 'total_losses', 'approx_kl']
        super().__init__(self.keys)


class RDQNLogger(Logger):
    """A Rainbow DQN logger that stores information for each episode iteration."""
    def __init__(self) -> None:
        self.keys = ['returns', 'actions', 'train_losses', 'double_q_values',
                     'double_q_probs', 'env_info', 'ep_scores']
        super().__init__(self.keys)
