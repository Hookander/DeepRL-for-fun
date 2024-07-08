"""``RepeatAction`` wrapper - The chosen action will be reapeted n times"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium import Wrapper
import numpy as np


__all__ = ["DetectDeathV0"]


class DetectDeathV0(Wrapper):

    def __init__(
        self, env: gym.Env[ObsType, ActType], penalty: float = -1):
        """Initialize DetectDeath wrapper.

        Args:
            env (Env): the wrapped environment
        
        Because the gym environment is an emulation from the original atari game,
        the reward is the score given by the game... not very useful for a RL agent.
        We only gain points by shooting the aliens, so we need to detect when the player ship is destroyed.
        
        For that we will use the bottom part of the screen, where the player ship is. 
        Each time the ship is destroyed, a yellow number appears on the screen,
        so we need to detect this yellow color to know when the ship is destroyed, and give a negative reward.
        Note that because the number of lives is showed at the beginning of the game, 
        the agent will always gain negative rewards at the beginning of the game. (doesnt change anything however)
        
        The color is (in RGB) (162, 134, 56)
        
        For each action, we check a small square of the screen to detect the color.
        
        """
        super().__init__(env)
        assert penalty < 0, "Penalty must be negative"
        self.penalty = penalty
        
        self.x1 = 90
        self.x2 = 93
        self.y1 = 187
        self.y2 = 188

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

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, dict[str, Any]]:
        """Take a step in the environment."""
        state, reward, term, trunc, info = self.env.step(action)

        square_to_check = state[self.y1:self.y2, self.x1:self.x2]
        
        if self.check_square(square_to_check):
            reward += self.penalty

        return state, reward, term, trunc, info