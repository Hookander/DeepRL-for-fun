import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import nb_from_space
from networks.base_net import BaseNet

class CNN(BaseNet):

    def __init__(self, observation_space, action_space, config):
        super(CNN, self).__init__()

        self.n_observations = nb_from_space(observation_space)
        self.n_actions = nb_from_space(action_space)

        if len(observation_space.shape) == 1:
            raise ValueError("Observation space is not an image")

        self.maxpool = nn.MaxPool2d(3)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Conv2d(3, 16, kernel_size=3)
        self.layer2 = nn.Conv2d(16, 32, kernel_size=3)
        self.layer3 = nn.Conv2d(32, 64, kernel_size=3)
        self.layer4 = nn.Linear(64*2*2, 256)
        self.layer5 = nn.Linear(256, self.n_actions)


    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # from NHWC to NCHW
        #print(x.shape)
        x = F.relu(self.layer1(x))
        x = self.maxpool(x)
        x = F.relu(self.layer2(x))
        x = self.maxpool(x)
        x = F.relu(self.layer3(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)

        return x