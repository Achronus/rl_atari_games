import gym
from gym.wrappers import (
    FrameStack,
    GrayScaleObservation,
    ResizeObservation,
    RecordVideo,
    RecordEpisodeStatistics
)

from core.parameters import EnvParameters


class EnvDetails:
    """
    A basic class that contains an OpenAI Gym's environment details.

    Parameters:
        params (EnvParameters) - a data class containing environment parameters
    """
    def __init__(self, params: EnvParameters) -> None:
        assert not params.env_name == '', "'env_name' must contain an environment name!"
        self.gym_name = params.env_name
        self.name = self.gym_name.split('-')[0].split('/')[-1]

        self.env: gym.Env = None
        self.obs_space: gym.Space = None
        self.action_space: gym.Space = None
        self.input_shape: tuple = None
        self.n_actions: int = 0

        self.img_size = params.img_size
        self.stack_size = params.stack_size
        self.capture_video = params.capture_video
        self.record_every = params.record_every
        self.seed = params.seed

        self.__set()

    def make_env(self) -> gym.Env:
        """Makes a gym environments with multiple wrappers."""
        env = gym.make(self.gym_name)
        env = RecordEpisodeStatistics(env)

        if self.capture_video:
            env = RecordVideo(env, "videos", episode_trigger=lambda t: t % self.record_every == 0)

        env = ResizeObservation(env, shape=self.img_size)  # default image dim: [128, 128]
        env = GrayScaleObservation(env, keep_dim=False)  # Grayscale images
        env = FrameStack(env, num_stack=self.stack_size)  # default: 4 frames at a time

        # Set seed for environment
        env.seed(self.seed)
        env.action_space.seed(self.seed)
        env.observation_space.seed(self.seed)
        return env

    def __set(self) -> None:
        """
        Sets the OpenAI Gym environment to the class instance. Passes the environment through three wrappers:
        image grey scaling, image resizing, and frame stacking.
        """
        env = self.make_env()

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
            'stack_size': self.stack_size,
            'capture_video': self.capture_video,
            'record_every': self.record_every
        }
        return f'{attribute_dict}'
