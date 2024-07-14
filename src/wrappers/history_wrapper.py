"""
    A Wrapper to gives the models a sense of history.
"""

import numpy as np
from gymnasium import ObservationWrapper
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from collections import deque


class HistoryWrapper(ObservationWrapper):
    def __init__(self, env, n_history=4, **kwargs):
        super().__init__(env)
        self.n_history = n_history
        self.queue = deque([], maxlen=n_history)
        
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(*self.observation_space.shape[:-1], self.observation_space.shape[-1] * n_history),
            dtype=np.uint8
        )

    def observation(self, obs):
        self.queue.append(obs)
        while len(self.queue) < self.n_history:
            self.queue.append(np.zeros_like(obs))
        return np.concatenate(self.queue, axis=-1)