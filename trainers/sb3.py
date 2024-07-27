import torch
import torch.optim as optim
import math
import gymnasium as gym
from networks.linear import *
from src.utils import *
from src.wrappers import *
from itertools import count
import torch
from typing import Dict, Type
from trainers.base_trainer import BaseTrainer
from networks.base_net import BaseNet
import wandb
import os
import yaml
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

class Sb3Trainer(BaseTrainer):

    def __init__(self, config : Dict, network : Type[BaseNet]):

        super().__init__(config)

        self.networkClass = network
        self.config_trainer = config['trainers']
        self.config_network = config['networks']
        
        self.env_name = config['env']
        
        #self.number_of_repeats = self.config['number_of_repeats']
        
        self.do_wandb = config['do_wandb']
        self.wandb_config = config['wandb_config']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.is_continuous = self.config['continuous']

        self.env = self.get_env()

        if self.do_wandb:
            wandb.init(project=self.wandb_config['project'], config = self.config, monitor_gym=True)


    def get_env(self):
        """
        Returns the environment with the wrappers applied
        The wrapper dict is in the config file {wrapper_name(Str): wrapper_params(Dict)}
        """
        
        try :
            # Not all environements can be continuous, so we need to handle this case
            self.env = gym.make(self.env_name, continuous = self.is_continuous)
        except:
            self.env = gym.make(self.env_name)
        
        # Applies the wrappers
        wrapper_dict = self.config['wrappers']
        
        for wrapper_name, wrapper_params in wrapper_dict.items():
            WrapperClass = wrapper_name_to_WrapperClass[wrapper_name]
            if WrapperClass in compatible_wrappers[self.env_name]:
                self.env = WrapperClass(self.env, **wrapper_params)

        
        return self.env
    
    def train(self):
        
        self.env = make_atari_env(self.env_name, n_envs=4, seed=0)
        self.env = VecFrameStack(self.env, n_stack=4)
        self.env = VecVideoRecorder(self.env, "data/videos/" + str(wandb.run.name), record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
        self.model = PPO("CnnPolicy", self.env, verbose=1)
        self.model.learn(total_timesteps=10, callback=WandbCallback(verbose = 2))

    def save_model(self):
        path = "data/models/"
        path = path + str(wandb.run.name)
        os.mkdir(path)
        model_path = path + "/model"
        config_path = path + "/config.yaml"
        self.model.save(model_path)
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file)
