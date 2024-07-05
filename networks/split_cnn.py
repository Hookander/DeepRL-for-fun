import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import nb_from_space
from networks.base_net import BaseNet


"""
    A Network designed to work with the CarRacing-v2 environment.
    The image give as input contains 2 elements:
     - The track with the car and the road 
     - The HUD with the speed, RPM, etc at the botton of the image.
    With this network, we separate the two and feed it to 2 separate cnns,
    so they would both have to focus on 1 job each.
"""

class SplitCNN(BaseNet):

    def __init__(self, observation_space, action_space, config):
        super(SplitCNN, self).__init__()

        self.n_observations = nb_from_space(observation_space)
        self.n_actions = nb_from_space(action_space)

        if len(observation_space.shape) == 1:
            raise ValueError("Observation space is not an image")

        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        
        # The road cnn
        self.layer1 = nn.Conv2d(3, 16, kernel_size=3)
        self.layer2 = nn.Conv2d(16, 32, kernel_size=3)
        self.layer3 = nn.Conv2d(32, 64, kernel_size=3)
        self.layer4 = nn.Conv2d(64, 128, kernel_size=3)
        
        self.road_cnn = nn.Sequential(
            self.layer1,
            self.maxpool,
            self.relu,
            self.layer2,
            self.maxpool,
            self.relu,
            self.layer3,
            self.maxpool,
            self.relu,
            self.layer4,
            self.maxpool,
            self.relu,
            self.flatten
        )
        
        
        self.layer4 = nn.Linear(64*2*2, 256)
        self.layer5 = nn.Linear(256, self.n_actions)

        self.split = 83 # The HUD is 13 pixel tall with an (96, 96, 3) image

    def forward(self, x):

        x = x.permute(0, 3, 1, 2) # from NHWC to NCHW

        # Split the image in 2
        x_road = x[:, :, :self.split, :]
        x_hud = x[:, :, self.split:, :]

        x_road = self.road_cnn(x_road)
        print(x_road.shape)
        """
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
        """

        return x