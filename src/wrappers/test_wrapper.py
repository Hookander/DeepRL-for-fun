"""``RepeatAction`` wrapper - The chosen action will be reapeted n times"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium import Wrapper
import numpy as np

class TestWrapper(Wrapper):

    def __init__(
        self, env: gym.Env[ObsType, ActType], **kwargs):
        print('TEST WRAPPER INIT')
        
        gym.Wrapper.__init__(self, env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment."""
        self.last_action = None

        return super().reset(seed=seed, options=options)


    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, dict[str, Any]]:
        """Take a step in the environment."""
        state, reward, term, trunc, info = self.env.step(action)
        #print('TEST WRAPPER STEP')

        return state, reward, term, trunc, info