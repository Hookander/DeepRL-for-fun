import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_net import BaseNet

class CNN(BaseNet):

    def __init__(self, n_observations, n_actions):
        super(CNN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 300)
        self.layer2 = nn.Linear(300, 300)
        self.layer3 = nn.Linear(300, 300)
        self.layer4 = nn.Linear(300, n_actions)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)