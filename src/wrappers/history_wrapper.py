import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from collections import deque


class RelativePosition(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.pile_state = deque([], maxlen = 4)

    def observation(self, obs):
        
        