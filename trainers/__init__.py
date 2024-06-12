from typing import Dict, Type
from trainers.base_trainer import BaseTrainer
from trainers.basic import BasicTrainer

trainer_name_to_TrainerClass : Dict[str, Type[BaseTrainer]] = {
    "basic" : BasicTrainer,
}