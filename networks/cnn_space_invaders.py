import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import nb_from_space
from networks.base_net import BaseNet


"""
    A Network designed to work with the ALE/SpaceInvaders-v5 environment.
    the input is a (210, 160, 3) image.
    
    We crop a little the image to remove the score and the bottom of the screen.
"""

class SpaceInvadersCNN(BaseNet):

    def __init__(self, observation_space, action_space, config):
        super(SpaceInvadersCNN, self).__init__()

        self.n_observations = nb_from_space(observation_space)
        self.n_actions = nb_from_space(action_space)

        if len(observation_space.shape) == 1:
            raise ValueError("Observation space is not an image")
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # We keep the middle
        self.split1 = 22
        self.split2 = 210 - 15

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        
        """The road cnn"""
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3)
        self.layer2 = nn.Conv2d(32, 50, kernel_size=3)
        self.layer3 = nn.Conv2d(50, 64, kernel_size=3)
        self.layer4 = nn.Conv2d(64, 128, kernel_size=3)
        self.layer5 = nn.Conv2d(128, 256, kernel_size=3)
        
        self.lin1 = nn.LazyLinear(600)
        self.lin2 = nn.Linear(600, 300)
        self.lin3 = nn.Linear(300, self.n_actions)
        
        self.model = nn.Sequential(
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
            nn.MaxPool2d(2),
            self.relu,
            self.layer5,
            nn.MaxPool2d(2),
            self.relu,
            self.flatten,
            self.lin1,
            self.relu,
            self.lin2,
            self.relu,
            self.lin3
            
        )
        
        # For the lazy linear layer
        self.dummy_init()
        
    def dummy_init(self):
        """
        The use of Lazy modules requires a dummy input to be passed through the network
        """
        x = torch.randn(self.observation_space.shape).unsqueeze(0)

        x = self.forward(x)
        
        
    
    def forward(self, x):

        x = x.permute(0, 3, 1, 2) # from NHWC to NCHW

        # Keep the middle
        x = x[:, :, self.split1:self.split2, :]

        x = self.model(x)
        
        return x