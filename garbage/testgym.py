import gymnasium as gym
from gymnasium.utils import play
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("ALE/SpaceInvaders-v5")

state, info = env.reset()

x1 = 90
x2 = 93
y1 = 187
y2 = 188

square = state[y1:y2, x1:x2]

#print(square)

show_states = []
j = 0

def check_square(square):
    for line in square:
        if [162, 134, 56] in line:
            return True
    return False
for i in range(1000):
    action = 4 #env.action_space.sample()
    state, reward, term, trunc, info = env.step(action)
    n_state = state[y1:y2, x1:x2]
    # detect 162 134 56
    if check_square(n_state):
        #print("DEATH")
        show_states.append(state)
        j+=1
    if term or trunc:
        print('END')
        break
    if j>100:
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


plt.plot([x1, x1], [y1, y2], color='red', linewidth=1)
plt.plot([x1, x2], [y2, y2], color='red', linewidth=1)
plt.plot([x2, x2], [y2, y1], color='red', linewidth=1)
plt.plot([x2, x1], [y1, y1], color='red', linewidth=1)
print(len(show_states))
# Extract the square from state and put it in a tab
square = state[y1:y2, x1:x2]
state = show_states[64]
plt.imshow(state)
plt.show()
env.close()