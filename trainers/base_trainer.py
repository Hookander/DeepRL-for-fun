import torch
import torch.optim as optim
import math
import gymnasium as gym
from typing import Dict, Type
from src.utils import *
from itertools import count
from abc import ABC, abstractmethod




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
    def save_model(self):
        """
        Saves the model
        """
        raise NotImplementedError
