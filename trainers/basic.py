import torch
import torch.optim as optim
import math
import gymnasium as gym
from networks.linear import *
from src.utils import *
from itertools import count
import torch
from typing import Dict, Type
from trainers.base_trainer import BaseTrainer
from networks.base_net import BaseNet
from src.wrappers.repeat_wrapper import RepeatActionV0
from src.wrappers.space_invaders.detect_death import DetectDeathV0
import wandb
import yaml



class BasicTrainer(BaseTrainer):

    def __init__(self, config : Dict, network : Type[BaseNet]):

        super().__init__(config)

        self.networkClass = network
        self.config_trainer = config['trainers']
        self.config_network = config['networks']

        print(self.config_trainer)
        self.env_name = config['env']
        self.number_of_repeats = self.config['number_of_repeats']

        self.num_episodes = self.config_trainer['num_episodes'] #1000
        self.batch_size = self.config_trainer['batch_size']
        self.gamma = self.config_trainer['gamma'] #0.99
        self.tau = self.config_trainer['tau'] #0.001
        self.lr = self.config_trainer['learning_rate'] #0.0001
        self.eps_start = self.config_trainer['epsilon_start'] #0.9
        self.eps_end = self.config_trainer['epsilon_end'] #0.01
        self.eps_decay = self.config_trainer['epsilon_decay'] #0.995
        

        self.do_wandb = config['do_wandb']
        self.wandb_config = config['wandb_config']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.is_continuous = self.config['continuous']

        try :
            # Not all environements can be continuous, so we need to handle this case
            self.env = gym.make(self.env_name, continuous = self.is_continuous)
        except:
            self.env = gym.make(self.env_name)
        
        self.env = RepeatActionV0(self.env, self.number_of_repeats)
        self.env = DetectDeathV0(self.env, penalty = -1)

        self.policy_net = network(self.env.observation_space, self.env.action_space, self.config_network).to(self.device)
        self.target_net = network(self.env.observation_space, self.env.action_space, self.config_network).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

        if self.do_wandb:
            wandb.init(project=self.wandb_config['project'], config = self.config)
            wandb.watch(self.policy_net, log="all")
            wandb.config.update({"model_architecture": str(self.policy_net)})



    def select_action(self, state : torch.Tensor):
        
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device)
            #return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        out = self.policy_net(state_batch)
        state_action_values = out.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    def train(self):

        for i_episode in range(self.num_episodes):
            total_reward = 0
            # Initialize the environment and get its state
            state, info = self.env.reset()
            
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            for t in count():
                action = self.select_action(state)
                # print(action.flatten().numpy())
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                total_reward += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                #print(terminated, truncated, done)
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    break
            if self.do_wandb:
                wandb.log({'total_reward': total_reward, 'episode': i_episode, 'epsilon': self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)})
        print("training done")
