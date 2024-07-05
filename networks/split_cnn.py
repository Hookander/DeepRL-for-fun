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
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.split = 83 # The HUD is 13 pixel tall with an (96, 96, 3) image

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        
        """The road cnn"""
        self.layer1 = nn.Conv2d(3, 16, kernel_size=3)
        self.layer2 = nn.Conv2d(16, 32, kernel_size=3)
        self.layer3 = nn.Conv2d(32, 64, kernel_size=3)
        self.layer4 = nn.Conv2d(64, 128, kernel_size=3)
        
        self.road_lin1 = nn.LazyLinear(300)
        
        self.road_cnn = nn.Sequential(
            self.layer1,
            nn.MaxPool2d(2),
            self.relu,
            self.layer2,
            nn.MaxPool2d(2),
            self.relu,
            self.layer3,
            nn.MaxPool2d(2),
            self.relu,
            self.layer4,
            nn.MaxPool2d(3),
            self.relu,
            self.flatten,
            self.road_lin1,
            self.relu
        )
        
        """The HUD cnn"""
        
        self.hud1 = nn.Conv2d(3, 16, kernel_size=3)
        self.hud2 = nn.Conv2d(16, 32, kernel_size=3)
        
        self.hud_lin1 = nn.LazyLinear(300)
        
        self.hud_cnn = nn.Sequential(
            self.hud1,
            nn.MaxPool2d(2),
            self.relu,
            self.hud2,
            nn.MaxPool2d(2),
            self.relu,
            self.flatten,
            self.hud_lin1,
            self.relu
        )
        
        """The final layers"""
        
        self.final_lin1 = nn.LazyLinear(200)
        self.final_lin2 = nn.Linear(200, self.n_actions)
        
        self.final_net = nn.Sequential(
            self.final_lin1,
            self.relu,
            self.final_lin2
        )
        
        self.dummy_init()
        
    def dummy_init(self):
        """
        The use of Lazy modules requires a dummy input to be passed through the network
        """
        x = torch.randn(self.observation_space.shape).unsqueeze(0).permute(0, 3, 1, 2)
        
        x_road = x[:, :, :self.split, :]
        x_hud = x[:, :, self.split:, :]
        x_road = self.road_cnn(x_road)
        x_hud = self.hud_cnn(x_hud)
        x = torch.cat((x_road, x_hud), dim=1)
        x = self.final_net(x)
        
        
    
    def forward(self, x):

        x = x.permute(0, 3, 1, 2) # from NHWC to NCHW

        # Split the image in 2
        x_road = x[:, :, :self.split, :]
        x_hud = x[:, :, self.split:, :]

        x_road = self.road_cnn(x_road)
        #print(x_road.shape)
        x_hud = self.hud_cnn(x_hud)
        #print(x_hud.shape)
        
        x = torch.cat((x_road, x_hud), dim=1)
        x = self.final_net(x)

        return x