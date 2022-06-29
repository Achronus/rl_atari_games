import os
import tarfile
from typing import Union

from agents.dqn import DQN
from agents.rainbow import RainbowDQN
from agents.ppo import PPO
from core.parameters import ModelParameters
from core.enums import CoreCheckpointParams
from models._base import BaseModel
from utils.logger import Logger

import torch


class DataLoader:
    """A class dedicated to loading model data."""
    def __init__(self, filename: str, device: str) -> None:
        self.filename = filename
        self.device = device
        self.cp_data = self.get_checkpoint_data()

    def load_dqn_model(self) -> DQN:
        """Load a DQN model's parameters from the given filename. Files must be stored within a saved_models folder."""
        dqn = DQN(self.cp_data['env_details'], self.cp_data['model_params'], self.cp_data['params'],
                  device=self.device, seed=self.cp_data['seed'])
        dqn.local_network.load_state_dict(self.cp_data['other'].get('local_network'), strict=False)
        dqn.target_network.load_state_dict(self.cp_data['other'].get('target_network'), strict=False)
        return dqn

    def load_rdqn_model(self) -> RainbowDQN:
        """Load a DQN model's parameters from the given filename. Files must be stored within a saved_models folder."""
        rdqn = RainbowDQN(self.cp_data['env_details'], self.cp_data['model_params'], self.cp_data['params'],
                          self.cp_data['other'].get('buffer_params'), device=self.device, seed=self.cp_data['seed'])
        rdqn.local_network.load_state_dict(self.cp_data['other'].get('local_network'), strict=False)
        rdqn.target_network.load_state_dict(self.cp_data['other'].get('target_network'), strict=False)
        return rdqn

    def load_ppo_model(self) -> PPO:
        """Load a PPO model's parameters from the given filename. Files must be stored within a saved_models folder."""
        ppo = PPO(self.cp_data['env_details'], self.cp_data['model_params'], self.cp_data['params'],
                  device=self.device, seed=self.cp_data['seed'])
        ppo.network.load_state_dict(self.cp_data['other'].get('network'), strict=False)
        return ppo

    def get_checkpoint_data(self) -> dict:
        """Gets the checkpoint data, creates the respective objects and return the info as a dictionary."""
        checkpoint = torch.load(f'saved_models/{self.filename}.pt', map_location=self.device)
        env_details = checkpoint.get('env_details')
        core_keys = [item.value for item in CoreCheckpointParams]

        return {
            CoreCheckpointParams.ENV_DETAILS.value: env_details,
            CoreCheckpointParams.PARAMS.value: checkpoint.get('params'),
            CoreCheckpointParams.SEED.value: checkpoint.get('seed'),
            CoreCheckpointParams.MODEL_PARAMS.value: ModelParameters(
                network=BaseModel(input_shape=env_details.input_shape, n_actions=env_details.n_actions),
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
