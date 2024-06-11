# Logging
import os
import wandb

# Config system
import hydra
from omegaconf import OmegaConf, DictConfig


# ML libraries
import random
import numpy as np


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: DictConfig):
    print("Configuration used :")
    print(OmegaConf.to_yaml(config))
    config = OmegaConf.to_container(config, resolve=True)

if __name__ == "__main__":
    main()
