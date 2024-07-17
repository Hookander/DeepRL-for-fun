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
        return action, dist.log_prob(action)
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class ParallelizedPPO(BaseTrainer): 
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
        self.num_env = config['number_of_environments']
        self.is_continuous = self.config['continuous']
        
        self.do_wandb = config['do_wandb']
        self.wandb_config = config['wandb_config']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env, self.envs = self.get_env()
        
        self.net = network(self.env.observation_space, self.env.action_space, self.config_network).to(self.device)
        
        self.policy_net = ActorCritic(self.net).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        self.policy_old = ActorCritic(self.net).to(self.device)
        self.policy_old.load_state_dict(self.policy_net.state_dict())
        
        if self.do_wandb:
            wandb.init(project=self.wandb_config['project'], config=self.config)
        
    def get_env(self):
        wrapper_dict = self.config['wrappers']
        wrappers_lambda = []
        selected_wrappers = []
        def make_wrapped_env():
            env = gym.make(self.env_name)
            for wrapper_name, wrapper_params in wrapper_dict.items():
                WrapperClass = wrapper_name_to_WrapperClass[wrapper_name]
                if WrapperClass in compatible_wrappers[self.env_name]:
                    env = WrapperClass(env, **wrapper_params)
                    selected_wrappers.append(wrapper_name)
            return env

        self.envs = gym.vector.AsyncVectorEnv([make_wrapped_env for _ in range(self.num_env)])
        self.env = make_wrapped_env()
        
        return self.env, self.envs
    
    def optimize_model(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards_full = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        old_states_full = torch.stack(memory.states).detach()
        old_actions_full = torch.tensor(memory.actions, dtype=torch.long).detach()
        old_logprobs_full = torch.stack(memory.logprobs).detach()
        
        for _ in range(self.K_epochs):
            
            dataset = torch.utils.data.TensorDataset(old_states_full, old_actions_full, old_logprobs_full, rewards_full)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            for old_states, old_actions, old_logprobs, rewards in dataloader:
            
                logprobs, state_values, dist_entropy = self.policy_net.evaluate(old_states.to(self.device), old_actions.to(self.device))
                
                ratios = torch.exp(logprobs - old_logprobs.detach())
                advantages = rewards - state_values.detach()
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                loss = -torch.min(surr1, surr2) + 0.5*nn.MSELoss()(state_values, rewards) - 0.01*dist_entropy
                
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy_net.state_dict())
    
    def train(self):
        memory = Memory()
        
        time_step = 0
        total_reward = 0
        epis_0_endings = 0
        cpt = 0
        
        states, _ = self.envs.reset()
        states = torch.FloatTensor(states).to(self.device)
        
        for t in count():
            time_step += 1
            
            actions, log_probs = self.policy_old.act(states)
            next_states, rewards, terminateds, truncateds, _ = self.envs.step(actions.cpu().numpy())
            
            total_reward += rewards[0]  # Log reward for the first environment
            
            memory.states.extend(states)
            memory.actions.extend(actions)
            memory.logprobs.extend(log_probs)
            memory.rewards.extend(torch.FloatTensor(rewards).to(self.device))
            memory.is_terminals.extend(torch.BoolTensor(terminateds).to(self.device))
            
            states = torch.FloatTensor(next_states).to(self.device)
            
            if time_step % self.update_timestep == 0:
                self.optimize_model(memory)
                memory.clear_memory()
                time_step = 0
            
            if terminateds[0] or truncateds[0]:
                if self.do_wandb:
                    wandb.log({'total_reward': total_reward, 'episode': epis_0_endings})
                epis_0_endings += 1
                total_reward = 0
                
            cpt += list(terminateds).count(True) + list(truncateds).count(True)
            if self.do_wandb:
                wandb.log({'Episode endings' : cpt})
            if cpt >= self.num_episodes:
                break
        
        print("Training done")