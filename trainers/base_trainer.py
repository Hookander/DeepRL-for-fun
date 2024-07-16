import torch
import torch.optim as optim
import math
import gymnasium as gym
from typing import Dict, Type
from src.utils import *
from itertools import count
from abc import ABC, abstractmethod
import os
import wandb
import yaml




class BaseTrainer(ABC):

    def __init__(self, config : Dict):

        self.config = config

    @abstractmethod
    def select_action(self, state):
        
        """
        Returns an action based on the state
        """
        raise NotImplementedError

    @abstractmethod
    def optimize_model(self):
        """
        Optimizes the model (policy network)
        """
        raise NotImplementedError
    
    @abstractmethod
    def train(self):
        """
        Trains the model
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_env(self):
        """
        Returns the environment with the wrappers applied
        """
        raise NotImplementedError

    def save_model(self, path = "data/models/"):
        if self.do_wandb:
            path = path + str(wandb.run.name) + '_r'
            os.mkdir(path)
            config_path = path + "/config.yaml"
            model_path = path + "/model.pth"
            with open(config_path, 'w') as file:
                yaml.dump(self.config, file)
            torch.save(self.policy_net.state_dict(), model_path)
            wandb.save(config_path)
            wandb.save(model_path)

        else:
            torch.save(self.policy_net.state_dict(), path + self.env_name + '_policy_net.pth')
        print('Model saved at : ' + path + self.env_name + '_policy_net.pth')
