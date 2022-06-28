import os
import random
import sys
from typing import Union
import numpy as np
import yaml

from agents.dqn import DQN
from agents.rainbow import RainbowDQN
from agents.ppo import PPO
from core.env_details import EnvDetails
from core.enums import OptionalParams, ValidModels, ValidIMMethods
from core.exceptions import MissingVariableError
from core.parameters import (
    BufferParameters,
    DQNParameters,
    EnvParameters,
    PPOParameters,
    ModelParameters,
    RainbowDQNParameters
)
from intrinsic.parameters import IMParameters
from models._base import BaseModel
from models.actor_critic import ActorCritic
from models.dueling import CategoricalNoisyDueling
from utils.helper import dict_search

import torch
import torch.optim as optim


def create_model(model_type: str, env: str = 'primary', device: str = None,
                 filename: str = 'parameters', im_type: str = None) -> Union[DQN, RainbowDQN, PPO]:
    """
    Initializes predefined parameters from a yaml file and creates a model of the specified type.
    Returns the model as a class instance.

    :param model_type (str) - name of the model ('dqn', 'ppo' or 'rainbow')
    :param env (str) - an optional parameter that defines a custom environment to use
    :param device (str) - an optional parameter that defines a CUDA device to use
    :param filename (str) - the YAML filename in the root directory that contains hyperparameters
    :param im_type (str) - the name of the intrinsic motivation to use ('curiosity', 'empowerment' or 'surprise_based')
    """
    valid_names = list(ValidModels.__members__.keys())
    valid_im_names = list(ValidIMMethods.__members__.keys())
    name = model_type.lower()
    if name not in valid_names:
        raise ValueError(f"Model type '{model_type}' does not exist! Must be one of: {valid_names}.")
    if im_type is not None and im_type.lower() not in valid_im_names:
        raise ValueError(f"IM type '{im_type}' does not exist! Must be one of: {valid_im_names}.")

    # Get yaml file parameters
    params = YamlParameters(filename)

    # Check parameters are valid
    CheckParamsValid(name, params, im_type)

    # Create selected model
    set_model = SetModels(name, params, env, device, im_type)
    return set_model.create()


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
                available_params.append(param_name.lower())
        return available_params

    def __repr__(self) -> str:
        return f'Available attributes: {self.keys}'


class CheckParamsValid:
    """
    A class that checks if the required parameters are available in the yaml file.

    :param model_type (str) - name of the model ('dqn', 'ppo' or 'rainbow')
    :param yaml_params (YamlParameters) - YAML parameters extracted from a file, stored in a class object
    :param im_type (str) - name of the type of intrinsic motivation method to use
    """
    core_optim = ['lr', 'eps']
    core_agent = ['gamma', 'update_steps', 'clip_grad']

    def __init__(self, model_type: str, yaml_params: YamlParameters, im_type: str = None) -> None:
        self.available_params = yaml_params.available_params
        self.im_type = im_type

        self.env_params = self.get_attribute_names(EnvParameters)
        self.core_params = self.core_optim + self.core_agent
        self.dqn_params = self.get_attribute_names(DQNParameters)
        self.ppo_params = self.get_attribute_names(PPOParameters)
        self.rainbow_params = self.get_attribute_names(RainbowDQNParameters)
        self.buffer_params = self.get_attribute_names(BufferParameters)

        # Set desired parameters
        params = self.core_params + self.env_params
        if model_type == 'dqn':
            params += self.dqn_params
        elif model_type == 'rainbow':
            params += self.rainbow_params + self.buffer_params
        elif model_type == 'ppo':
            params += self.ppo_params

        # Add intrinsic motivation parameters
        if self.im_type is not None:
            self.im_params = self.__get_im_parameters()
            params += self.im_params

        # Handle optional parameters
        optional_params = list(OptionalParams.__members__.keys())
        params = [key for key in params if key not in optional_params]

        # Check parameters exist
        unique_params = list(set(params))
        self.check_params(unique_params)

    def __get_im_parameters(self) -> list:
        """Gets the data class associated to the given intrinsic motivation method."""
        # Handle name formatting
        name = self.im_type.title()
        name = ''.join(name.split('_'))

        # Get the parameters class object
        im_params_class_object = getattr(sys.modules['intrinsic.parameters'], f'{name}Parameters')
        params = self.get_attribute_names(im_params_class_object)
        return params

    def check_params(self, param_list: list) -> None:
        """Checks if the given parameters are set in the yaml file."""
        # Handle missing parameters
        false_bools = [idx for idx, item in enumerate(param_list) if item not in self.available_params]
        if len(false_bools) >= 1:
            missing_params = [param_list[i] for i in false_bools]
            raise MissingVariableError(f"Cannot find variables {missing_params} in yaml file! Have you added them? "
                                       f"Refer to 'core/template.yaml' for the required format.")

    @staticmethod
    def get_attribute_names(cls) -> list:
        """Gets a list of attribute names for a given class."""
        keys = list(cls.__dict__.keys())
        return [key.lower() for key in keys if "__" not in key]


class SetModels:
    """
    A class that sets the parameters from a predefined yaml file and creates model instances.

    :param model_type (str) - name of the model ('dqn', 'ppo' or 'rainbow')
    :param yaml_params (YamlParameters) - class object that contains parameters from a YAML file
    :param env (str) - an optional parameter that defines a custom environment to use. When set to
                       'primary', environment 'env_name' from 'yaml_params' is used
    :param device (str) - an optional parameter that defines a CUDA device to use
    :param im_type (str) - name of the type of intrinsic motivation method to use
    """
    def __init__(self, model_type: str, yaml_params: YamlParameters,
                 env: str = 'primary', device: str = None, im_type: str = None) -> None:
        self.model_type = model_type
        self.yaml_params = yaml_params
        self.seed = yaml_params.environment['seed']
        self.optim_params = yaml_params.core_optimizer
        self.dqn_core_params = {**self.yaml_params.core_agent, **self.yaml_params.dqn_core}
        self.device = device
        self.env = env
        self.im_params = None

        # Seeding
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env_details = self.__create_env_details()

        if im_type is not None:
            self.im_name = im_type
            self.im_params = self.__create_im_params(im_type)

    def create(self) -> Union[DQN, RainbowDQN, PPO]:
        """Create a model based on the given name."""
        # Handle intrinsic motivation method
        if self.im_params is not None:
            im_type = (self.im_name, self.im_params)
        else:
            im_type = None

        # Create class instance
        if self.model_type == 'dqn':
            return self.__create_dqn(im_type)
        elif self.model_type == 'rainbow':
            return self.__create_rainbow_dqn()
        elif self.model_type == 'ppo':
            return self.__create_ppo()

    def __create_im_params(self, im_type: str) -> IMParameters:
        """Creates a set of intrinsic motivation parameters based on the given type."""
        params = getattr(self.yaml_params, f'intrinsic_{im_type}')
        params = {**params, 'input_shape': self.env_details.input_shape,
                  'n_actions': self.env_details.n_actions}

        # Handle name formatting
        name = im_type.title()
        name = ''.join(name.split('_'))

        # Get the parameters class object
        class_object = getattr(sys.modules['intrinsic.parameters'], f'{name}Parameters')
        return class_object(**params)

    def __create_model_params(self, net_type: BaseModel, **net_kwargs) -> ModelParameters:
        """Creates a set of model parameters based on the given network type."""
        network = net_type(**net_kwargs)
        return ModelParameters(
            network=network,
            optimizer=optim.Adam(network.parameters(), lr=self.optim_params['lr'],
                                 eps=self.optim_params['eps'])
        )

    def __create_env_details(self) -> EnvDetails:
        """Creates an environment details class."""
        if self.env != 'primary':
            self.yaml_params.environment['env_name'] = self.env

        env_params = EnvParameters(**self.yaml_params.environment)
        return EnvDetails(env_params)

    def __create_dqn(self, im_type: tuple) -> DQN:
        """Creates a DQN model from predefined parameters."""
        core_params = {key: val for key, val in self.dqn_core_params.items() if key not in ['clip_grad']}
        buffer_params = {'buffer_size': self.yaml_params.dqn_buffer['buffer_size'],
                         'batch_size': self.yaml_params.dqn_buffer['batch_size']}
        dqn_params = {**core_params, **self.yaml_params.dqn_vanilla, **buffer_params}
        params = DQNParameters(**dqn_params)

        model_params = self.__create_model_params(
            BaseModel,
            input_shape=self.env_details.input_shape,
            n_actions=self.env_details.n_actions
        )
        return DQN(self.env_details, model_params, params, self.device, self.seed, im_type)

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
        return RainbowDQN(self.env_details, model_params, params, buffer_params,
                          self.device, self.seed)

    def __create_ppo(self) -> PPO:
        """Creates a PPO model from predefined parameters."""
        model_params = self.__create_model_params(
            ActorCritic,
            input_shape=self.env_details.input_shape,
            n_actions=self.env_details.n_actions
        )

        ppo_params = {**self.yaml_params.core_agent, **self.yaml_params.ppo}
        params = PPOParameters(**ppo_params)
        return PPO(self.env_details, model_params, params, self.device, self.seed)


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
