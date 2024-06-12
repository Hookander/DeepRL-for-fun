import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseNet(nn.Module, ABC):

    def __init__(self):
        super(BaseNet, self).__init__()

    
    @abstractmethod
    def forward(self, x):
        """
        Returns the output of the model
        """

        raise NotImplementedError