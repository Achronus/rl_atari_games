import os
import random
from typing import Union
import numpy as np
import yaml

from agents.dqn import DQN, RainbowDQN
from agents.ppo import PPO
from core.env_details import EnvDetails
from core.exceptions import MissingVariableError
from core.parameters import (
    BufferParameters,
    DQNParameters,
    EnvParameters,
    PPOParameters,
    ModelParameters,
    RainbowDQNParameters
)
from models._base import BaseModel
from models.actor_critic import ActorCritic
from models.cnn import CNNModel
from models.dueling import CategoricalNoisyDueling
from utils.helper import dict_search

import torch
import torch.optim as optim


def create_model(model_type: str, filename: str = 'parameters') -> Union[DQN, RainbowDQN, PPO]:
    """Initializes predefined parameters from a yaml file and creates a model of the specified type.
    Returns the model as a class instance."""
    valid_names = ['dqn', 'ppo', 'rainbow']
    name = model_type.lower()
    if name not in valid_names:
        raise ValueError(f"Model type '{model_type}' does not exist! Must be one of: {valid_names}.")

    # Get yaml file parameters
    params = YamlParameters(filename)

    # Check parameters are valid
    CheckParamsValid(name, params)

    # Create selected model
    set_model = SetModels(params)
    return set_model.create(name)


class YamlParameters:
    """A class for storing yaml file parameters. Available attributes are visible via 'print(instance_name)'."""
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.documents = self.get_parameters(filename)

        self.root_keys = self.set_root_key_values()
        self.child_keys = self.set_child_key_values()
        self.__tidy_attributes()

        self.keys = self.root_keys + self.child_keys
        self.available_params = self.__get_available_parameters()

    @staticmethod
    def get_parameters(filename: str) -> list:
        """Returns a list of documents from a given filename. File must be stored in the root directory."""
        f = filename if filename[-5:] == '.yaml' else f'{filename}.yaml'
        assert os.path.exists(f), f"'{f}' does not exist! Have you created it in '{os.getcwd()}'?"

        stream = open(f"{f}", 'r')
        return list(yaml.load_all(stream, yaml.Loader))

    def set_root_key_values(self) -> list:
        """Adds root key values to the class instance and returns a list of root keys."""
        root_keys = []
        for document in self.documents:
            for root_key, values in document.items():
                setattr(self, root_key, values)
                root_keys.append(root_key)
        return root_keys

    def set_child_key_values(self) -> list:
        """Adds child key values to the class instance based on stored root keys. Returns a list of child keys."""
        child_keys = []
        for root_key in self.root_keys:
            child_dict = getattr(self, root_key)

            for child_key, values in child_dict.items():
                if isinstance(values, dict):
                    key_name = f'{root_key}_{child_key}'
                    setattr(self, key_name, values)
                    child_keys.append(key_name)
        return child_keys

    def __tidy_attributes(self) -> None:
        """Removes root key attributes if they consist of child keys."""
        root_removal = []
        # Get root keys containing dictionaries
        for key in self.child_keys:
            root_key = key.split('_')  # [root_key, child_key]
            root_removal.append(root_key[0])

        # Remove above keys and attributes
        root_removal = set(root_removal)
        for key in root_removal:
            self.root_keys.remove(key)
            delattr(self, key)

    def __get_available_parameters(self) -> list:
        """Obtains the parameter names from the root and child keys and returns them as a list."""
        available_params = []
        for key in self.keys:
            params = getattr(self, key)

            for param_name in params.keys():
                available_params.append(param_name)
        return available_params

    def __repr__(self) -> str:
        return f'Available attributes: {self.keys}'


class CheckParamsValid:
    """A class that checks if the required parameters are available in the yaml file."""
    core_optim = ['lr', 'eps']
    core_agent = ['gamma', 'update_steps', 'clip_grad']

    def __init__(self, model_type: str, yaml_params: YamlParameters) -> None:
        self.available_params = yaml_params.available_params

        self.env_params = self.get_attribute_names(EnvParameters)
        self.core_params = self.core_optim + self.core_agent
        self.dqn_params = self.get_attribute_names(DQNParameters)
        self.ppo_params = self.get_attribute_names(PPOParameters)
        self.rainbow_params = self.get_attribute_names(RainbowDQNParameters)
        self.buffer_params = self.__get_buffer_keys()

        # Set desired parameters
        params = self.core_params + self.env_params
        if model_type == 'dqn':
            params += self.dqn_params
        elif model_type == 'rainbow':
            params += self.rainbow_params + self.buffer_params
        elif model_type == 'ppo':
            params += self.ppo_params

        # Check parameters exist
        unique_params = list(set(params))
        self.check_params(unique_params)

    def __get_env_keys(self) -> list:
        """Gets the environment parameters attribute names as a list and updates certain keys.
        Returns the updated list."""
        keys = self.get_attribute_names(EnvParameters)

        updated_keys = [key.replace(key, 'ENV_1') if key == 'ENV_NAME' else key for key in keys]  # ENV_NAME -> ENV_1
        updated_keys = [key for key in updated_keys if key not in ['RECORD_EVERY', 'SEED']]  # Remove keys
        return updated_keys

    def __get_buffer_keys(self) -> list:
        """Gets the buffer parameters attribute names as a list and updates certain keys.
                Returns the updated list."""
        keys = self.get_attribute_names(BufferParameters)

        updated_keys = [key for key in keys if key not in ['input_shape']]  # Remove keys
        return updated_keys

    def check_params(self, param_list: list) -> None:
        """Checks if the given parameters are set in the yaml file."""
        # Handle missing parameters
        false_bools = [idx for idx, item in enumerate(param_list) if item not in self.available_params]
        if len(false_bools) >= 1:
            missing_params = [param_list[i] for i in false_bools]
            raise MissingVariableError(f"Cannot find variables {missing_params} in yaml file! Have you added them? "
                                       f"Refer to 'core/template.yaml' for required format.")

    @staticmethod
    def get_attribute_names(cls) -> list:
        """Gets a list of attribute names for a given class."""
        keys = list(cls.__dict__.keys())
        return [key for key in keys if "__" not in key]


class SetModels:
    """A class that sets the parameters from a predefined yaml file and creates model instances."""
    def __init__(self, yaml_params: YamlParameters) -> None:
        self.yaml_params = yaml_params
        self.seed = yaml_params.environment['seed']
        self.optim_params = yaml_params.core_optimizer
        self.dqn_core_params = {**self.yaml_params.core_agent, **self.yaml_params.dqn_core}

        # Seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env_details = self.__create_env_details()

    def create(self, model_name: str) -> Union[DQN, RainbowDQN, PPO]:
        """Create a model based on the given name."""
        if model_name == 'dqn':
            return self.__create_dqn()
        elif model_name == 'rainbow':
            return self.__create_rainbow_dqn()
        elif model_name == 'ppo':
            return self.__create_ppo()

    def __create_model_params(self, net_type: BaseModel, **net_kwargs) -> ModelParameters:
        """Creates a set of model parameters based on the given network type."""
        network = net_type(**net_kwargs)
        return ModelParameters(
            network=network,
            optimizer=optim.Adam(network.parameters(), lr=self.optim_params['lr'], eps=self.optim_params['eps'])
        )

    def __create_env_details(self) -> EnvDetails:
        """Creates an environment details class."""
        env_params = EnvParameters(**self.yaml_params.environment)
        return EnvDetails(env_params)

    def __create_dqn(self) -> DQN:
        """Creates a DQN model from predefined parameters."""
        core_params = {key: val for key, val in self.dqn_core_params.items() if key not in ['clip_grad']}
        dqn_params = {**core_params, **self.yaml_params.dqn_vanilla}
        params = DQNParameters(**dqn_params)

        model_params = self.__create_model_params(
            CNNModel,
            input_shape=self.env_details.input_shape,
            n_actions=self.env_details.n_actions
        )
        return DQN(self.env_details, model_params, params, self.seed)

    def __create_rainbow_dqn(self) -> RainbowDQN:
        """Creates a Rainbow DQN model from predefined parameters."""
        rdqn_params = {**self.dqn_core_params, **self.yaml_params.dqn_rainbow}
        params = RainbowDQNParameters(**rdqn_params)
        buffer_params = BufferParameters(**self.yaml_params.dqn_buffer, input_shape=self.env_details.input_shape)

        model_params = self.__create_model_params(
            CategoricalNoisyDueling,
            input_shape=self.env_details.input_shape,
            n_actions=self.env_details.n_actions,
            n_atoms=params.n_atoms
        )
        return RainbowDQN(self.env_details, model_params, params, buffer_params, self.seed)

    def __create_ppo(self) -> PPO:
        """Creates a PPO model from predefined parameters."""
        model_params = self.__create_model_params(
            ActorCritic,
            input_shape=self.env_details.input_shape,
            n_actions=self.env_details.n_actions
        )

        ppo_params = {**self.yaml_params.core_agent, **self.yaml_params.ppo}
        params = PPOParameters(**ppo_params)
        return PPO(self.env_details, model_params, params, self.seed)


def get_utility_params(filename: str = 'parameters') -> dict:
    """Returns a dictionary containing utility variables, obtained from a given yaml file."""
    params = YamlParameters(filename)
    template_params = YamlParameters('core/template.yaml')

    util_params = list(set(params.available_params) - set(template_params.available_params))
    flat_dict = {key: val for d in params.documents for key, val in d.items()}
    util_dict = {}
    for param in util_params:
        for key, value in dict_search(flat_dict):
            if key == param:
                util_dict[key] = value
    return util_dict
