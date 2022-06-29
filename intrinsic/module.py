from abc import abstractmethod

from intrinsic.parameters import (
    IMParameters,
    IMExperience,
    CuriosityParameters,
    EmpowermentParameters,
    SurpriseBasedParameters
)
from intrinsic.model import CuriosityModel, EmpowermentModel, SurpriseBasedModel
from models._base import BaseModel

import torch


class IMMethod:
    """A parent class for intrinsic motivation loss calculations."""
    def __init__(self, params: IMParameters, model: BaseModel, device: str) -> None:
        self.params = params
        self.model = model
        self.device = device

    @abstractmethod
    def get_loss(self, experience):
        pass

    @abstractmethod
    def compute_loss(self, experience, model_loss):
        pass

    @abstractmethod
    def compute_return(self, experience):
        pass


class Curiosity(IMMethod):
    """
    Simulates the curiosity intrinsic reward as displayed in the Curiosity-driven Exploration paper:
    https://arxiv.org/abs/1705.05363.
    """
    def __init__(self, params: CuriosityParameters, model: CuriosityModel, device: str):
        super().__init__(params, model, device)

    def get_loss(self, experience: IMExperience) -> tuple:
        """Computes the required loss values unique to curiosity."""
        encoded_state = self.model.encode(experience.state)
        encoded_next_state = self.model.encode(experience.next_state)

        # Compute state and action predictions
        pred_states = self.model.forward(encoded_state.detach(), experience.actions.detach())
        pred_actions = self.model.predict_action(experience.state, experience.next_state)

        # Calculate state and action errors
        actions = experience.actions.detach().flatten()
        forward_loss = self.params.forward_loss(pred_states, encoded_next_state.detach()).sum(dim=1).unsqueeze(1)
        inverse_loss = self.params.inverse_loss(pred_actions, actions).unsqueeze(1)
        return forward_loss, inverse_loss

    def compute_loss(self, experience: IMExperience, model_loss: torch.Tensor) -> tuple:
        """Computes the total loss using the curiosity and model losses."""
        forward_loss, inverse_loss = self.get_loss(experience)
        curiosity_loss = (1 - self.params.comparison_weight) * inverse_loss.flatten().mean()
        curiosity_loss += self.params.comparison_weight * forward_loss.flatten().mean()
        curiosity_loss = curiosity_loss.sum() / curiosity_loss.flatten().shape[0]

        loss = curiosity_loss + model_loss.mean()
        return loss, curiosity_loss

    def compute_return(self, experience: IMExperience) -> torch.Tensor:
        """Computes the intrinsic reward signal for curiosity."""
        forward_loss, _ = self.get_loss(experience)
        reward = (1. / self.params.scaling_factor) * forward_loss
        return reward.detach()


class Empowerment(IMMethod):
    """
    Simulates the empowerment intrinsic reward as displayed in the X paper:

    """
    def __init__(self, params: EmpowermentParameters, module: EmpowermentModel, device: str):
        super().__init__(params, module, device)

    def get_loss(self, experience):
        """Computes the required loss values unique to empowerment."""
        pass

    def compute_loss(self, experience, model_loss):
        """Computes the total loss using the empowerment and model losses."""
        pass

    def compute_return(self, experience: IMExperience) -> torch.Tensor:
        """Computes the intrinsic reward signal for empowerment."""
        pass


class SurpriseBased(IMMethod):
    """
    Simulates the surprise-based intrinsic reward as displayed in the X paper:

    """
    def __init__(self, params: SurpriseBasedParameters, module: SurpriseBasedModel, device: str):
        super().__init__(params, module, device)

    def get_loss(self, experience):
        """Computes the required loss values unique to surprise-based motivation."""
        pass

    def compute_loss(self, experience, model_loss):
        """Computes the total loss using the surprise-based and model losses."""
        pass

    def compute_return(self, experience: IMExperience) -> torch.Tensor:
        """Computes the intrinsic reward signal for surprised-based motivation."""
        pass
