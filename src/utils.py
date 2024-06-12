from collections import namedtuple, deque
import random
from gymnasium.spaces import Discrete, Box


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def nb_from_space(space):
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Box):
        nb = 1
        for dim in space.shape:
            nb *= dim

        return nb
    else:
        raise ValueError("Space not recognized")