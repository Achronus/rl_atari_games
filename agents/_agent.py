from typing import Protocol


class Agent(Protocol):
    """A base class for Reinforcement Learning agents."""
    def act(self) -> int:
        """Returns actions for a given state based on its current policy."""
        pass

    def learn(self) -> None:
        """Updates the network parameters."""
        pass
