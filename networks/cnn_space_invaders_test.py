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

class SpaceInvadersCNNTest(BaseNet):

    def __init__(self, observation_space, action_space, config):
        super(SpaceInvadersCNNTest, self).__init__()

        self.n_observations = nb_from_space(observation_space)
        self.n_actions = nb_from_space(action_space)

        if len(observation_space.shape) == 1:
            raise ValueError("Observation space is not an image")
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.model = nn.Linear(1, 1)
        self.dummy_init()
        
    def dummy_init(self):
        """
        The use of Lazy modules requires a dummy input to be passed through the network
        """
        x = torch.randn(self.observation_space.shape).unsqueeze(0)

        x = self.forward(x)
        
        
    
    def forward(self, x):

        return torch.randn(self.n_actions)

