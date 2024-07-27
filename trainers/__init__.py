from typing import Dict, Type
from trainers.base_trainer import BaseTrainer
from trainers.basic import BasicTrainer
from trainers.parallelized_dqn import Parallelized_DQN
from trainers.ppo import PPO
from trainers.parallelized_ppo import ParallelizedPPO
from trainers.sb3 import Sb3Trainer

trainer_name_to_TrainerClass : Dict[str, Type[BaseTrainer]] = {
    "basic" : BasicTrainer,
    "parallel" : Parallelized_DQN,
    "ppo" : PPO,
    "parallel_ppo" : ParallelizedPPO,
    "sb3" : Sb3Trainer
}