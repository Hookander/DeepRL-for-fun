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

class SpaceInvadersCNN(BaseNet):

    def __init__(self, observation_space, action_space, config):
        super(SpaceInvadersCNN, self).__init__()

        self.n_observations = nb_from_space(observation_space)
        self.n_actions = nb_from_space(action_space)

        if len(observation_space.shape) == 1:
            raise ValueError("Observation space is not an image")
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # To crop the image and split it in 2 
        self.cnn_min_h = 22
        self.cnn_max_h = 183
        self.ship_min_h = 194
        self.ship_max_h = 195

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        
        """The alien cnn"""
        self.alien_out = 500
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3)
        self.layer2 = nn.Conv2d(32, 50, kernel_size=3)
        self.layer3 = nn.Conv2d(50, 64, kernel_size=3)
        self.layer4 = nn.Conv2d(64, 128, kernel_size=3)
        self.layer5 = nn.Conv2d(128, 256, kernel_size=3)
        
        self.lin1 = nn.LazyLinear(800)
        self.lin2 = nn.Linear(800, 500)
        self.lin3 = nn.Linear(500, 500)
        
        self.alien_cnn = nn.Sequential(
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
            self.lin3,
            self.relu
        )
        
        # The player's ship network
        # The ship beeing green, we only keep the green channel
        self.player_out = 30
        self.player1 = nn.Linear(160, 200)
        self.player2 = nn.Linear(200, 200)
        self.player3 = nn.Linear(200, 30)
        
        self.player_net = nn.Sequential(
            self.flatten,
            self.player1,
            self.relu,
            self.player2,
            self.relu,
            self.player3,
            self.relu
        )
        
        #final network
        self.fin1 = nn.LazyLinear(800)
        self.fin2 = nn.Linear(800, 500)
        self.fin3 = nn.Linear(500, self.n_actions)
        
        self.final_net = nn.Sequential(
            self.fin1,
            self.relu,
            self.fin2,
            self.relu,
            self.fin3
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

        x_alien = x[:, :, self.cnn_min_h:self.cnn_max_h, :]
        x_ship = x[:, 0, self.ship_min_h:self.ship_max_h, :]
        
        x_alien = self.alien_cnn(x_alien)
        x_ship = self.player_net(x_ship)
        
        x = torch.cat((x_alien, x_ship), 1)
        x = self.final_net(x)
        
        return x