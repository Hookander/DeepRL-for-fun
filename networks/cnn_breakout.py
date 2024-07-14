import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import nb_from_space
from networks.base_net import BaseNet


"""
    A Network designed to work with the ALE/SpaceInvaders-v5 environment.
    the input is a (210, 160, 3) image.
    
    We crop a little the image to remove the score and the bottom of the screen.
    We also separate to image in 2 parts, the top and the bottom (with the player ship).
    One network will be used for each part.
    The goal of the bottom network is to detect th eplayer ship, so we only need to keep the green channel,
    but we also only need 1 line of thickness. This allows us to use a very simple network, not a 2d cnn.
    
    Note that the reward is very basic (points for shooting the aliens) but  can't be modified,
    so there is no sanction for losing a life for example... Because of that performances are not very good.
"""

class BreakoutCNN(BaseNet):

    def __init__(self, observation_space, action_space, config):
        super(BreakoutCNN, self).__init__()
        
        #observation space = Box(0, 255, (210, 160, 3), uint8)

        self.n_observations = nb_from_space(observation_space)
        self.n_actions = nb_from_space(action_space)

        if len(observation_space.shape) == 1:
            raise ValueError("Observation space is not an image")
        
        
        self.cnn1 = nn.Conv2d(4, 32, kernel_size=3)
        self.cnn2 = nn.Conv2d(32, 50, kernel_size=3)
        self.cnn3 = nn.Conv2d(50, 64, kernel_size=3)
        self.cnn4 = nn.Conv2d(64, 128, kernel_size=3)
        
        self.flatten = nn.Flatten()
        
        self.cnn = nn.Sequential(
            self.cnn1,
            nn.MaxPool2d(2),
            nn.ReLU(),
            self.cnn2,
            nn.MaxPool2d(2),
            nn.ReLU(),
            self.cnn3,
            nn.MaxPool2d(2),
            nn.ReLU(),
            self.cnn4,
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        
        self.lin1 = nn.LazyLinear(1500)
        self.lin2 = nn.Linear(1500, 800)
        self.lin3 = nn.Linear(800, 500)
        self.lin4 = nn.Linear(500, nb_from_space(action_space))
        
        self.model = nn.Sequential(
            self.cnn,
            self.flatten,
            self.lin1,
            nn.ReLU(),
            self.lin2,
            nn.ReLU(),
            self.lin3,
            nn.ReLU(),
            self.lin4
        )
        
        # For the lazy linear layer
        self.dummy_init()
        
    def dummy_init(self):
        """
        The use of Lazy modules requires a dummy input to be passed through the network
        """
        x = torch.randn((210, 160, 4)).unsqueeze(0)

        x = self.forward(x)
        
        
    
    def forward(self, x):

        x = x.permute(0, 3, 1, 2) # from NHWC to NCHW


        x = self.model(x)
        
        return x