# Logging
import os
import wandb

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig

# ML libraries
import random
import numpy as np

# Project files

from networks import network_name_to_ModelClass
from trainers import trainer_name_to_TrainerClass


@hydra.main(config_path="configs", config_name="config_default.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)

    trainer_name = config['trainer']
    network_name = config['network']

    ModelClass = network_name_to_ModelClass[network_name]
    TrainerClass = trainer_name_to_TrainerClass[trainer_name]


    trainer = TrainerClass(config, ModelClass)

    #trainer.train()
    #trainer.save_model()

if __name__ == "__main__":
    main()
