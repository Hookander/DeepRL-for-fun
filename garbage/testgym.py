import gymnasium as gym
from gymnasium.utils import play
import matplotlib.pyplot as plt

env = gym.make('CarRacing-v2', continuous = False)

state, info = env.reset()
for i in range(50):
    action = 0 #env.action_space.sample()
    state, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break

plt.imshow(state)

d = 13
plt.plot([0, 96], [96-d, 96-d], color='red', linewidth=2)

plt.show()

env.close()