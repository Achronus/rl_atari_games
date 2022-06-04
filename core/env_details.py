from core.wrappers import ResizeObservation

import gym
from gym.wrappers import FrameStack, GrayScaleObservation


class EnvDetails:
    """
    A basic class that contains an OpenAI Gym's environment details.

    Parameters:
        gym_name (str) - a string of the OpenAI Gym environment name
        img_size (int) - a single integer used to resize the state space, defaults to 128
        stack_size (int) - a single integer of states (images) to pass per batch, defaults to 4
    """
    def __init__(self, gym_name: str, img_size: int = 128, stack_size: int = 4) -> None:
        assert isinstance(gym_name, str), f"Invalid 'env_names': {gym_name}. Expected a 'string'"
        assert isinstance(img_size, int), f"Invalid 'img_size': {img_size}. Expected type: 'int'"
        assert isinstance(stack_size, int), f"Invalid 'stack_size': {stack_size}. Expected type: 'int'"

        self.gym_name = gym_name
        self.name = gym_name.split('-')[0].split('/')[-1]
        self.env: gym.Env = None

        self.obs_space: gym.Space = None
        self.action_space: gym.Space = None
        self.input_shape: tuple = (0, 0, 0)
        self.n_actions: int = 0

        self.img_size = img_size
        self.stack_size = stack_size

        if gym_name != '':
            self.__set()

    def __set(self) -> None:
        """
        Sets the OpenAI Gym environment to the class instance. Passes the environment through three wrappers:
        image grey scaling, image resizing, and frame stacking.
        """
        env = GrayScaleObservation(gym.make(self.gym_name), keep_dim=False)  # Grayscale images
        env = ResizeObservation(env, shape=self.img_size)  # default image dim: [128, 128]
        env = FrameStack(env, num_stack=self.stack_size)  # default: 4 frames at a time

        self.env = env
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        self.input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n

    def __repr__(self):
        attribute_dict = {
            'gym_name': self.gym_name,
            'name': self.name,
            'env': self.env,
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'input_shape': self.input_shape,
            'n_actions': self.n_actions,
            'img_size': self.img_size,
            'stack_size': self.stack_size
        }
        return f'{attribute_dict}'
