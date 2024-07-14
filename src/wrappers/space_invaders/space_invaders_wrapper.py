"""``RepeatAction`` wrapper - The chosen action will be reapeted n times"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Box
from gymnasium import Wrapper
import numpy as np
from collections import deque

class SpaceInvadersWrapper(Wrapper):

    def __init__(
        self, env: gym.Env[ObsType, ActType], death_penalty: float = -1, missile_penalty:float = -1, **kwargs):
        print('SpaceInvadersWrapper init')
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
        gym.Wrapper.__init__(self, env)
        # The observation space is a 4d tensor (210, 160, 3) but we will convert it to grayscale
        # and concatenate the last 4 states to have a 4d tensor (210, 160, 4)
        self.observation_space = Box(0, 255, (210, 160, 4))
                
        
        assert death_penalty < 0, "death_penalty must be negative"
        assert missile_penalty < 0, "missile_penalty must be negative"
        
        self.missile_penalty = missile_penalty
        self.death_penalty = death_penalty
        
        self.x1 = 90
        self.x2 = 93
        self.y1 = 187
        self.y2 = 188
        self.check_reset_square = False
        
        self.player_color = [50, 132, 50]
        
        
        ##To check the missiles USELESS MISSILES NOT RENDERED IN THE STATE (don't know why)
        self.__ship_top = 184
        self.__missile_color = [142, 142, 142]
        self.area_height = 30 #24
        # To compare with the next area and check if a missile is coming or leaving
        self.previous_missile_first_line = None
        

        
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
    
    
    def get_pos(self, state):
        #The color of the ship is [50 132  50], always on the line 193
        line = state[193]
        args = np.argwhere(np.all(line == self.player_color, axis=-1))
        if len(args) > 1:
            left, right = args[0][0], args[-1][0]
            return left, right
        else: #player not on screen
            return None, None
    
    def check_missiles(self, state):
        # For now we don't check whether the missile is coming from the player
        # or the aliens
        current_missile_first_line = None
        left, right = self.get_pos(state)
        if left is None:
            self.previous_missile_first_line = None
            return False
        area = state[self.__ship_top - self.area_height:self.__ship_top, left:right]
        print(area[0])
        for n, line in enumerate(area):
            if self.__missile_color in line:
                current_missile_first_line = n
                break
        print(self.previous_missile_first_line, current_missile_first_line)
        if current_missile_first_line is None:
            self.previous_missile_first_line = None
            # In this case we can't know for sure if a missile is coming or leaving
            return False
        # We compare with the previous first line to check if a missile is coming or leaving
        
        if self.previous_missile_first_line is not None:
            if current_missile_first_line > self.previous_missile_first_line:
                self.previous_missile_first_line = current_missile_first_line
                print("Incoming missile detected")
                return True
            else:
                self.previous_missile_first_line = current_missile_first_line
                return False
        self.previous_missile_first_line = current_missile_first_line
        return False

    def change_reward_distribution(self, reward):
        """
        The environment gives a different reward depending on the line of the shot
        alien (the further the alien, the more points you get). This causes the agent to 
        try to predict the movements of the farthest aliens, which is not what we want.
        So we will change the reward
        
        According to the atari documentation, the reward is as follows(for each row):
        row 1 : 5, row 2 : 10, row 3 : 15, row 4 : 20, row 5 : 25, row 6 : 30

        We try the opposite, so the agent will try to shoot the closest aliens.
        
        We also check for missiles in front of the player ship.
        
        """
        change_dict = {0 : 0, 5: 30, 10: 25, 15: 20, 20: 15, 25: 10, 30: 5}
        if reward not in change_dict:
            print('Reward not in dict', reward)
            return reward
        return change_dict[reward]

    def check_death_penalty(self, state):
        ret = 0
        square_to_check = state[self.y1:self.y2, self.x1:self.x2]       
        if self.check_square(square_to_check) and self.check_reset_square == False:
            print("Penalty applied")
            ret = self.death_penalty
            
            # the lifes are showed for several frames, so we need to wait until the number disappears
            self.check_reset_square = True
        
        if self.check_reset_square:
            if not self.check_square(square_to_check):
                print("Lives disappeared")
                self.check_reset_square = False
        return ret

    def change_state(self, state):
        
        # Convert to grayscale
        state = np.dot(state[...,:3], [0.2989, 0.5870, 0.1140])
        state = np.expand_dims(state, axis=-1)
        return state

    def reset(self, **kwargs) -> ObsType:
        state, info = self.env.reset(**kwargs)
        state = self.change_state(state)
        return state, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, dict[str, Any]]:
        """Take a step in the environment."""
        state, reward, term, trunc, info = self.env.step(action)

        reward += self.check_death_penalty(state)
        
        # for now we just change the reward if there is a missile in front of the player ship
        # Later we will just give the info to the agent and let it learn by itself
        # reward += self.check_missiles(state) * self.missile_penalty
        
        state = self.change_state(state)
        return state, reward, term, trunc, info