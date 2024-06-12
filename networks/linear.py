import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import nb_from_space
from networks.base_net import BaseNet

class LinearNetwork(BaseNet):

    def __init__(self, observation_space, action_space, net_config):
        super(LinearNetwork, self).__init__()

        self.n_observations = nb_from_space(observation_space)
        self.n_actions = nb_from_space(action_space)

        self.flatten = nn.Flatten()

        layers = net_config['layers']

        self.model = nn.Sequential(
            self.flatten,
            nn.Linear(self.n_observations, layers[0]),
            nn.ReLU(),
        )
        for i in range(1, len(layers)):
            self.model.add_module(f'layer{i}', nn.Linear(layers[i-1], layers[i]))
            self.model.add_module(f'relu{i}', nn.ReLU())
        self.model.add_module('layer_end', nn.Linear(layers[-1], self.n_actions))

        
    def forward(self, x):
        return self.model(x)
