from models._base import BaseModel

import torch


class IMBaseModel(BaseModel):
    """A base intrinsic motivation (IM) model used across multiple IM methods."""
    def __init__(self, input_shape: tuple, n_actions: int, device: str) -> None:
        super().__init__(input_shape, n_actions)
        self.device = device
        self.conv_out_size = self.get_conv_size(input_shape)

    def one_hot_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Helper function that creates a matrix of one hot encoded action probabilities,
        where the action position = 1 and other values are 0."""
        one_hot_actions = torch.zeros(actions.shape[0], self.n_actions, device=self.device)
        indices = torch.stack((torch.arange(actions.shape[0], device=self.device), actions), dim=0).tolist()
        one_hot_actions[indices] = 1.
        return one_hot_actions  # shape -> (batch_size, n_actions)
