import torch
import torch.optim as optim
import math
import gymnasium as gym
import numpy as np
from networks.linear import *
from src.utils import *
from itertools import count
from typing import Dict, Type
from trainers.base_trainer import BaseTrainer
from networks.base_net import BaseNet
from src.wrappers.repeat_wrapper import RepeatActionV0
import wandb
import yaml
import os



class Parallelized_DQN(BaseTrainer):

    def __init__(self, config : Dict, network : Type[BaseNet]):

        super().__init__(config)

        self.networkClass = network
        self.config_trainer = config['trainers']
        self.config_network = config['networks']

        print(self.config_trainer)
        self.env_name = config['env']
        self.num_env = config['number_of_environments']
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

        #try :
            # Not all environements can be continuous, so we need to handle this case
        #    self.env = gym.make(self.env_name, continuous = self.is_continuous)
        #except:
        #    self.env = gym.make(self.env_name)

        self.envs = gym.make_vec(self.env_name, self.num_env)
        self.envs = RepeatActionV0(self.envs, self.number_of_repeats)

        # To get the observation aned action spaces
        self.env = gym.make(self.env_name)

        self.policy_net = network(self.env.observation_space, self.env.action_space, self.config_network).to(self.device)
        self.target_net = network(self.env.observation_space, self.env.action_space, self.config_network).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0

        if self.do_wandb:
            wandb.init(project=self.wandb_config['project'], config = self.config)



    def select_action(self, states : [torch.Tensor], running_env_mask : [int]):
        
        samples = [random.random() for _ in range(len(states))]
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        actions = [torch.tensor([0], device = self.device) for i in range(len(states))]
        for i, state in enumerate(states):
            if running_env_mask[i] == 1:
                if samples[i] > eps_threshold:
                    with torch.no_grad():
                        # t.max(1) will return the largest column value of each row.
                        # second column on max result is index of where max element was
                        # found, so we pick action with the larger expected reward.
                        actions[i] = self.policy_net(state).max(1).indices.view(1, 1)
                else:
                    actions[i] = torch.tensor([[self.env.action_space.sample()]], device=self.device)
        return actions

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
        """
        
        """

        total_reward = 0
        # Initialize the environment and get its state
        states, infos = self.envs.reset()
        running_env_mask = [1 for _ in range(self.num_env)]
        states = [torch.tensor(states[i], dtype=torch.float32, device=self.device).unsqueeze(0) for i in range(len(states))]
        cpt = 0
        for t in count():
            """
            #! problem : doesnt stop when next_state=None (bc all the environnements need to be terminated right know)
            -> call the model with None -> error

            -> we use a mask [], a 0 in pos i means the i-th environnement is terminated

            """
            actions = self.select_action(states, running_env_mask)

            actions_items = [actions[i].item() for i in range(len(actions))]
            observations, rewards, terminateds, truncateds, _ = self.envs.step(actions_items)

            #the envs that were done are reset
            #running_env_mask = [1 if mask == 0 or mask == 1 else 0 for mask in running_env_mask]

            total_reward += rewards[0] # we only log the reward of the first environnement (the others are the same)
            rewards = [torch.tensor([rewards[i]], device=self.device) for i in range(len(rewards))]

            # determine of all episodes are done (truncated or terminated)<
            next_states = []
            for i, observation in enumerate(observations):
                if terminateds[i]:
                    next_states.append(None)
                else:
                    next_states.append(torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))


            """
            Store the transitions in memory
            so we unwrap the states, actions, ...
            """
            for i in range(len(states)):
                if running_env_mask[i]:
                    self.memory.push(states[i], actions[i], next_states[i], rewards[i])
            """
            We update this after updating the memory to allow it to contain the final transistions 
            of some simulations, but by making sure those transisitons aren't re-added afterwards.
            """
            running_env_mask = [0 if terminateds[i] or truncateds[i] else 1 for i in range(len(terminateds))]


            # Move to the next state
            states = next_states

            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)


            if self.do_wandb and (terminateds[0] or truncateds[0]):
                wandb.log({'total_reward': total_reward})
                total_reward = 0
            cpt += sum(terminateds)

            if cpt >= self.num_episodes:
                break
        print("training done")
    
    def save_model(self, path = "data/models/"):
        if self.do_wandb:
            path = path + str(wandb.run.name)
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
