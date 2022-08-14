import os
import tarfile
from typing import Union

from agents._agent import Agent
from agents.dqn import DQN
from agents.rainbow import RainbowDQN
from agents.ppo import PPO
from core.create import create_model
from core.exceptions import MissingCheckpointKeyError, InvalidModelTypeError
from core.parameters import ModelParameters
from core.enums import CoreCheckpointParams, ValidIMMethods
from utils.logger import Logger

# Keep for globals() call in 'get_checkpoint_data()'
from models._base import BaseModel
from models.actor_critic import ActorCritic
from models.dueling import CategoricalNoisyDueling
from intrinsic.empower_models import PPONetwork, RainbowNetwork, QNetwork

import torch


class DataLoader:
    """A class dedicated to loading model data."""
    def __init__(self, filename: str, model_type: str, device: str) -> None:
        self.filename = filename
        self.model_type = model_type
        self.filepath = f'saved_models/{self.filename}'
        self.device = device
        self.cp_data = self.get_checkpoint_data()

    def load_dqn_model(self) -> DQN:
        """Load a DQN model's parameters from the given filename. Files must be stored within a saved_models folder."""
        params = self.__set_params('dqn')
        dqn = self.__valid_model_check(DQN, params)
        dqn.local_network.load_state_dict(self.cp_data['other'].get('local_network'), strict=False)
        dqn.target_network.load_state_dict(self.cp_data['other'].get('target_network'), strict=False)
        return dqn

    def load_rdqn_model(self) -> RainbowDQN:
        """Load a DQN model's parameters from the given filename. Files must be stored within a saved_models folder."""
        params = self.__set_params('rainbow')
        params.update(buffer_params=self.cp_data['other'].get('buffer_params'))
        rdqn = self.__valid_model_check(RainbowDQN, params)
        rdqn.local_network.load_state_dict(self.cp_data['other'].get('local_network'), strict=False)
        rdqn.target_network.load_state_dict(self.cp_data['other'].get('target_network'), strict=False)
        return rdqn

    def load_ppo_model(self) -> PPO:
        """Load a PPO model's parameters from the given filename. Files must be stored within a saved_models folder."""
        params = self.__set_params('ppo')
        ppo = self.__valid_model_check(PPO, params)
        ppo.network.load_state_dict(self.cp_data['other'].get('network'), strict=False)
        return ppo

    def __valid_model_check(self, agent: Agent, params: dict) -> Agent:
        try:
            return agent(**params)
        except AttributeError:
            raise InvalidModelTypeError(f"'{self.model_type}' does not match '{self.filepath}' architecture! "
                                        f"Are you using the correct file?")

    def __set_params(self, model_type: str) -> dict:
        """
        Sets the model parameters dictionary to be passed into the model to initialize it.

        :param model_type (string) - name of the model type. Options => ['ppo', 'rainbow', 'dqn']
        """
        im_type = self.cp_data['other']['im_type']

        if im_type is not None:
            model = create_model(model_type, device=self.device, im_type=im_type)
            im_controller = model.im_method

            if im_type == ValidIMMethods.EMPOWERMENT.value:
                cp_other = self.cp_data['other']
                im_controller.model.encoder.load_state_dict(cp_other.get('encoder'), strict=False)
                im_controller.model.source_net.load_state_dict(cp_other.get('source_net'), strict=False)
                im_controller.model.forward_net.load_state_dict(cp_other.get('forward_net'), strict=False)
                im_controller.model.source_target.load_state_dict(cp_other.get('source_target'), strict=False)
                im_controller.model.forward_target.load_state_dict(cp_other.get('forward_target'), strict=False)
            im_type = (im_type, im_controller)

        return dict(
            env_details=self.cp_data['env_details'],
            model_params=self.cp_data['model_params'],
            params=self.cp_data['params'],
            device=self.device,
            seed=self.cp_data['seed'],
            im_type=im_type
        )

    def get_checkpoint_data(self) -> dict:
        """Gets the checkpoint data, creates the respective objects and return the info as a dictionary."""
        checkpoint = torch.load(self.filepath, map_location=self.device)
        env_details = checkpoint.get('env_details')
        core_keys = [item.value for item in CoreCheckpointParams]

        try:
            model = globals()[checkpoint.get(CoreCheckpointParams.NETWORK_TYPE.value)]
        except KeyError:
            raise MissingCheckpointKeyError(f"'{CoreCheckpointParams.NETWORK_TYPE.value}' key is missing from "
                                            f"'{self.filepath}'! Are you using the correct file?")

        return {
            CoreCheckpointParams.ENV_DETAILS.value: env_details,
            CoreCheckpointParams.PARAMS.value: checkpoint.get('params'),
            CoreCheckpointParams.SEED.value: checkpoint.get('seed'),
            CoreCheckpointParams.MODEL_PARAMS.value: ModelParameters(
                network=model(input_shape=env_details.input_shape, n_actions=env_details.n_actions),
                optimizer=checkpoint.get('optimizer'),
                loss_metric=checkpoint.get('loss_metric')
            ),
            'other': {key: val for key, val in checkpoint.items() if key not in core_keys}
        }

    def unpack_logger_data(self, env_name: str) -> Union[Logger, None]:
        """A helper function to unpack compressed logger data stored in the 'saved_models' directory.
        Returns an instance of the logger type."""
        name = f"saved_models/{self.filename.split('_')[0]}_{env_name}_logger_data"
        folder = '/'.join(self.filename.split('/')[:-1])
        file_in = f"{name}.tar.gz"
        file_out = f"{name}.pt"

        if os.path.exists(file_in):
            # Extract file
            tar = tarfile.open(file_in, "r:gz")
            tar.extractall(f"saved_models/{folder}")
            tar.close()

            # Load file to logger
            data = torch.load(file_out, map_location=self.device)
            os.remove(file_out)  # Remove uncompressed file
            return data['logger']
        else:
            return None
