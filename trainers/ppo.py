import torch
import torch.optim as optim
import math
import gymnasium as gym
import numpy as np
from torch.distributions import Categorical
from networks.linear import *
from src.utils import *
from src.wrappers import *
from itertools import count
from typing import Dict, Type
from trainers.base_trainer import BaseTrainer
from networks.base_net import BaseNet
import wandb
import yaml
import os


class ActorCritic(nn.Module):
    def __init__(self, base_net):
        """
            The base net is the network that will be used to create the actor and critic networks
            It's the one given in the config files
        """
        super(ActorCritic, self).__init__()
        self.base_net = base_net
        
        self.actor = nn.Sequential(
            self.base_net,
            nn.Softmax(dim=-1))
        
        self.critic = nn.Sequential(
            self.base_net,
            nn.Linear(base_net.n_actions, 1)
        )
        
    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO(BaseTrainer): 
    def __init__(self, config, network: Type[BaseNet]):
        
        super().__init__(config)
        
        self.networkClass = network
        self.config_trainer = config['trainers']
        self.config_network = config['networks']
        
        self.num_episodes = self.config_trainer['num_episodes']
        self.batch_size = self.config_trainer['batch_size']
        self.lr = self.config_trainer['learning_rate']
        self.gamma = self.config_trainer['gamma']
        self.eps_clip = self.config_trainer['eps_clip']
        self.K_epochs = self.config_trainer['K_epochs']
        self.update_timestep = self.config_trainer['update_timestep']

        self.env_name = config['env']
        self.is_continuous = self.config['continuous'] # For some envs
        
        self.do_wandb = config['do_wandb']
        self.wandb_config = config['wandb_config']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = self.get_env()
        
        self.net = network(self.env.observation_space, self.env.action_space, self.config_network).to(self.device)
        
        self.policy = ActorCritic(self.net).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        self.policy_old = ActorCritic(self.net).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        
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
    
    def optimize_model(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.tensor(memory.actions, dtype=torch.long).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states.to(self.device), old_actions)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5*nn.MSELoss()(state_values, rewards) - 0.01*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def train(self):
        
        memory = Memory()
        
        time_step = 0
        for i_episode in range(self.num_episodes):
            total_reward = 0
            
            state, info = self.env.reset()
            for t in count():
                time_step += 1
                
                action, log_prob = self.policy_old.act(torch.FloatTensor(state).unsqueeze(0).to(self.device))
                state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                memory.states.append(torch.FloatTensor(state))
                memory.actions.append(action)
                memory.logprobs.append(log_prob)
                memory.rewards.append(reward)
                memory.is_terminals.append(terminated or truncated)
                
                if time_step % self.update_timestep == 0:
                    self.optimize_model(memory)
                    memory.clear_memory()
                    time_step = 0
            
                if terminated or truncated:
                    break
            if self.do_wandb:
                wandb.log({'total_reward': total_reward, 'episode': i_episode})