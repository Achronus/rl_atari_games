from abc import abstractmethod

from intrinsic.parameters import (
    IMParameters,
    IMExperience,
    CuriosityParameters,
    EmpowermentParameters,
)
from intrinsic.model import CuriosityModel, EmpowermentModel
from models._base import BaseModel

import torch
import torch.nn as nn


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

        loss = curiosity_loss + model_loss
        return loss, curiosity_loss

    def compute_return(self, experience: IMExperience) -> torch.Tensor:
        """Computes the intrinsic reward signal for curiosity."""
        forward_loss, _ = self.get_loss(experience)
        reward = (1. / self.params.curiosity_weight) * forward_loss
        return reward.detach()


class Empowerment(IMMethod):
    """
    Simulates the empowerment intrinsic reward as displayed in the Empowerment-driven Exploration
    using Mutual Information Estimation paper: https://arxiv.org/abs/1810.05533.
    """
    def __init__(self, params: EmpowermentParameters, module: EmpowermentModel, device: str):
        super().__init__(params, module, device)
        self.soft_plus = nn.Softplus(beta=params.softplus_beta)
        self.n_actions = self.model.n_actions

        # Overwritten in IMController
        self.source_optim = None
        self.forward_optim = None

    def __compute_marginals(self, encoded_state: torch.Tensor) -> torch.Tensor:
        """Helper function for computing state marginals using all actions and the current state."""
        all_actions = torch.arange(self.model.n_actions).to(self.device)
        marginals = []

        # Iterate over each state in the batch
        for state in encoded_state:
            marginal_state = state.expand(len(all_actions), -1)  # Add action dimension

            # Predict next state and add to existing state
            with torch.no_grad():
                next_state_pred = self.model.forward_net(marginal_state, all_actions)
            next_state_pred = next_state_pred.detach() + state

            # Compute mean over actions for each state pixel
            marginal_pred = next_state_pred.mean(dim=0).unsqueeze(0)

            # Add state means to list, mean shape -> (1, state_size)
            marginals.append(marginal_pred)

        # Merge state means into shape -> (batch_size, state_size)
        return torch.cat(marginals)

    def get_loss(self, experience: IMExperience) -> torch.Tensor:
        """Gets the components for computing the loss values unique to empowerment."""
        # Compute state difference between current and prediction
        next_state_diff = self.model.forward_net(experience.state, experience.actions)
        next_state_pred = experience.state + next_state_diff
        forward_loss = self.params.state_loss(next_state_pred, experience.next_state)

        self.forward_optim.zero_grad()
        forward_loss.backward()
        self.forward_optim.step()

        return forward_loss

    def __get_action_preds(self, experience: IMExperience) -> tuple:
        """Compute the action predictions for the next state actions and the state marginals."""
        one_hot_actions = self.model.one_hot_actions(experience.actions)

        # Compute marginals
        state_marginals = self.__compute_marginals(experience.state)
        next_state_actions = torch.cat((experience.next_state, one_hot_actions), dim=1)
        marginal_state_actions = torch.cat((state_marginals, one_hot_actions), dim=1)

        # Get predicted actions, shape -> (batch_size, 1)
        action_pred = self.model.source_net(next_state_actions)
        next_action_pred = self.model.source_net(marginal_state_actions)
        return action_pred, next_action_pred

    def compute_loss(self, experience: IMExperience, model_loss: torch.Tensor) -> tuple:
        """Computes the total loss using the empowerment and model losses."""
        # Compute losses
        action_pred, next_action_pred = self.__get_action_preds(experience)
        source_loss = -((-self.soft_plus(-action_pred)).mean() - self.soft_plus(next_action_pred).mean())
        forward_loss = self.get_loss(experience)

        # Update source network
        self.source_optim.zero_grad()
        source_loss.backward()
        self.source_optim.step()

        # Set empowerment loss and total loss
        empowerment_loss = [forward_loss.item(), source_loss.item()]
        loss = model_loss
        return loss, empowerment_loss

    def compute_return(self, experience: IMExperience) -> torch.Tensor:
        """Computes the intrinsic reward signal for empowerment."""
        # Compute KL Divergence
        action_pred, next_action_pred = self.__get_action_preds(experience)
        mutual_info = -self.soft_plus(-action_pred) - self.soft_plus(next_action_pred)

        reward = self.params.empower_weight * mutual_info.detach()
        return reward.detach()
