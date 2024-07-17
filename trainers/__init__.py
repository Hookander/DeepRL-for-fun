from typing import Dict, Type
from trainers.base_trainer import BaseTrainer
from trainers.basic import BasicTrainer
from trainers.parallelized_dqn import Parallelized_DQN
from trainers.ppo import PPO
from trainers.parallelized_ppo import ParallelizedPPO

trainer_name_to_TrainerClass : Dict[str, Type[BaseTrainer]] = {
    "basic" : BasicTrainer,
    "parallel" : Parallelized_DQN,
    "ppo" : PPO,
    "parallel_ppo" : ParallelizedPPO
}