import gymnasium as gym
from gymnasium.utils import play
import matplotlib.pyplot as plt

env = gym.make("ALE/SpaceInvaders-v5")

state, info = env.reset()
for i in range(300):
    action = 4 #env.action_space.sample()
    state, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break

#plt.imshow(state)

d = 194
plt.plot([0, 160], [d, d], color='red', linewidth=1)

d = 195
plt.plot([0, 160], [d, d], color='red', linewidth=1)

d = 22
plt.plot([0, 160], [d, d], color='red', linewidth=1)

d = 183
plt.plot([0, 160], [d, d], color='red', linewidth=1)


plt.imshow(state[:,:,2])
plt.show()
env.close()