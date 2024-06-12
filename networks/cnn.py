import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import nb_from_space
from networks.base_net import BaseNet

class CNN(BaseNet):

    def __init__(self, observation_space, action_space):
        super(CNN, self).__init__()

        self.n_observations = nb_from_space(observation_space)
        self.n_actions = nb_from_space(action_space)

        if len(observation_space.shape) == 1:
            raise ValueError("Observation space is not an image")

        self.layer1 = nn.Linear(self.n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, self.n_actions)


    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)