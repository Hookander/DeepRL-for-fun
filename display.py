import torch as nn
import gymnasium as gym
import torch
from networks.linear import *
from networks.cnn import *
from networks.split_cnn import *
from networks.cnn_space_invaders import *
from networks.cnn_breakout import *
from src.wrappers.repeat_wrapper import RepeatActionV0
from src.wrappers.space_invaders.space_invaders_wrapper import SpaceInvadersWrapper
from src.wrappers import *
import pygame



class Displayer():

    def __init__(self, path_to_model):
        self.env_name = "ALE/Breakout-v5"
        self.env = gym.make(self.env_name, render_mode = 'human')
        self.env = RepeatActionV0(self.env, 0)
        self.env = BreakoutWrapper(self.env)
        self.env = HistoryWrapper(self.env, 4)

        observation_space, action_space = self.env.observation_space, self.env.action_space
        self.model = BreakoutCNN(observation_space, action_space, None)
        self.model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    
    def display(self, nb):
        for i in range(nb):
            state, info = self.env.reset()
            
            done = False
            while not done:
                state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
                self.env.render()
                with torch.no_grad():
                    action = self.model(state).max(1).indices.view(1, 1)
                state, reward, done, truncated, _ = self.env.step(action.item())
                if pygame.key.get_pressed()[pygame.K_SPACE]:
                    done = True
        self.env.close()

disply = Displayer('data/models/grateful-firebrand-157/model.pth')
disply.display(400)
