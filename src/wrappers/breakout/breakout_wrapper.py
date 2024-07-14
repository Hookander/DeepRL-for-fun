"""``RepeatAction`` wrapper - The chosen action will be reapeted n times"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box
from gymnasium import Wrapper
import numpy as np
from collections import deque

class BreakoutWrapper(Wrapper):

    def __init__(
        self, env: gym.Env[ObsType, ActType], death_penalty: float = -1, missile_penalty:float = -1, **kwargs):
        print('BreakoutWrapper init')
        
        gym.Wrapper.__init__(self, env)
        # The observation space is a 4d tensor (210, 160, 3) but we will convert it to grayscale
        # and concatenate the last 4 states to have a 4d tensor (210, 160, 4)
        self.observation_space = Box(0, 255, (210, 160, 4))

        self.state_pile = deque([], maxlen=4) # To concatenate the last 4 states
        

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment."""
        self.last_action = None

        return super().reset(seed=seed, options=options)

    def check_square(self, square):
        for line in square:
            if [162, 134, 56] in line:
                return True
        return False

    def change_state(self, state):
        
        # Convert to grayscale
        state = np.dot(state[...,:3], [0.2989, 0.5870, 0.1140])
        state = np.expand_dims(state, axis=-1)
        self.state_pile.append(state)
        if len(self.state_pile) <4:
            return np.concatenate([state for _ in range(4)], axis=-1)
        return np.concatenate(self.state_pile, axis=-1)

    def reset(self, **kwargs) -> ObsType:
        state, info = self.env.reset(**kwargs)
        state = self.change_state(state)
        return state, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, dict[str, Any]]:
        """Take a step in the environment."""
        state, reward, term, trunc, info = self.env.step(action)
        
        # for now we just change the reward if there is a missile in front of the player ship
        # Later we will just give the info to the agent and let it learn by itself
        # reward += self.check_missiles(state) * self.missile_penalty
        
        state = self.change_state(state)
        return state, reward, term, trunc, info