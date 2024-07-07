import torch as nn
import gymnasium as gym
import torch
from networks.linear import *
from networks.cnn import *
from networks.split_cnn import *
from src.wrappers.repeat_wrapper import RepeatActionV0
import pygame



class Displayer():

    def __init__(self, path_to_model):
        self.env_name = "CarRacing-v2"
        self.env = gym.make(self.env_name, render_mode = 'human', continuous = False)
        self.env = RepeatActionV0(self.env, 0)

        observation_space, action_space = self.env.observation_space, self.env.action_space
        self.model = SplitCNN(observation_space, action_space, None)
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

disply = Displayer('data/models/easy-firefly-76/model.pth')
disply.display(400)
