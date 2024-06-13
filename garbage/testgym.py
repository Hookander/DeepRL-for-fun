import gymnasium as gym
from gymnasium.utils import play

env = gym.make('CarRacing-v2', continuous = False, render_mode="rgb_array")
play.play(env, keys_to_action={'q':1, 'd':2, 'z':3, 's':4})