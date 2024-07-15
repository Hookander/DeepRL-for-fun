import gymnasium as gym
from gymnasium.utils import play
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("ALE/Breakout-v5")

state, info = env.reset()

for i in range(20):
    action = 0 #env.action_space.sample()
    state, reward, term, trunc, info = env.step(action)
    # detect 162 134 56
    if term or trunc:
        print('END')
        break

#plt.imshow(state)

"""d = 194
plt.plot([0, 160], [d, d], color='red', linewidth=1)

d = 195
plt.plot([0, 160], [d, d], color='red', linewidth=1)

d = 22
plt.plot([0, 160], [d, d], color='red', linewidth=1)

d = 183
plt.plot([0, 160], [d, d], color='red', linewidth=1)"""

wall_crop = 8
state = state[30:196, wall_crop:160 - wall_crop, :]
plt.imshow(state)
plt.show()
env.close()