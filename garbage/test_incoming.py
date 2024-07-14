import gymnasium as gym
from gymnasium.utils import play
import matplotlib.pyplot as plt
import numpy as np
import pygame

pygame.init()
env = gym.make("ALE/SpaceInvaders-v5")

state, info = env.reset()

def get_pos(state):
    #The color of the ship is [50 132  50], always on the line 193
    line = state[193]
    args = np.argwhere(np.all(line == [50, 132, 50], axis=-1))
    left, right = args[0][0], args[-1][0]
    print(left, right)
    return left, right



for i in range(49):
    action = 0 #env.action_space.sample()
    state, reward, term, trunc, info = env.step(action)

    if term or trunc:
        print('END')
        break
    
    if pygame.key.get_pressed()[pygame.K_SPACE]:
        break



#plt.imshow(state)
"""
d = 193
plt.plot([0, 160], [d, d], color='red', linewidth=1)

left, right = get_pos(state)
#vertical lines
plt.plot([left, left], [184, 165], color='red', linewidth=1)
plt.plot([right, right], [184, 165], color='red', linewidth=1)

area = state[160:184, left:right]
print(area.shape)"""

state = np.dot(state[...,:3], [0.2989, 0.5870, 0.1140])
state = np.expand_dims(state, axis=-1)
concate = np.concatenate((state, state, state), axis=-1)
print(concate.shape)
plt.imshow(state)
plt.show()
env.close()