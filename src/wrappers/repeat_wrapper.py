"""``RepeatAction`` wrapper - The chosen action will be reapeted n times"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType


__all__ = ["RepeatActionV0"]


class RepeatActionV0(
    gym.ActionWrapper[ObsType, ActType, ActType], gym.utils.RecordConstructorArgs
):

    def __init__(
        self, env: gym.Env[ObsType, ActType], number_of_repeats: int
    ):
        """Initialize RepeatyAction wrapper.

        Args:
            env (Env): the wrapped environment
            number_of_repeats (int): number of steps to repeat the action
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, number_of_repeats=number_of_repeats
        )
        gym.ActionWrapper.__init__(self, env)

        self.number_of_repeats = number_of_repeats
        self.executed_repeats = 0
        self.last_action: ActType | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment."""
        self.last_action = None

        return super().reset(seed=seed, options=options)

    def action(self, action: ActType) -> ActType:
        """Execute the action."""

        if self.executed_repeats >= self.number_of_repeats:
            self.executed_repeats = 0
            self.last_action = action
            return action
        if (
            self.last_action is not None
            and self.executed_repeats < self.number_of_repeats
        ):
            action = self.last_action
            self.executed_repeats += 1


        self.last_action = action
        return action