import numpy as np
import cv2

import gym
from gym.spaces import Box


class ResizeObservation(gym.ObservationWrapper):
    """Resizes an environments state space to a given shape."""
    def __init__(self, env: gym.Env, shape: int) -> None:
        super().__init__(env)
        self.shape = (shape, shape)  # Example: [128, 128]

        # Change image shape
        obs_shape = self.shape + self.observation_space.shape[2:]

        # Update observation space
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation: np.array) -> np.array:
        """Resize the given observation (image) into its new shape and return it."""
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation
