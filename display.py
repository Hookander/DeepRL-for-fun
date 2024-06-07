import torch as nn
import gymnasium as gym
import torch
from dqn import *



class Displayer():

    def __init__(self, path_to_model):
        self.env_name = "CartPole-v1"
        self.env = gym.make(self.env_name, render_mode = 'human')
        self.n_act = self.env.action_space.n
        self.n_obs = len(self.env.reset()[0])

        self.model = DQN(self.n_obs, self.n_act)
        self.model.load_state_dict(torch.load(path_to_model))
    
    def display(self, nb):
        for i in range(nb):
            state, info = self.env.reset()
            
            done = False
            while not done:
                state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
                self.env.render()
                action = self.model(state).max(1).indices.view(1, 1)
                state, reward, done, truncated, _ = self.env.step(action.item())
        self.env.close()

disply = Displayer('data/models/policy_net.pth')
disply.display(400)