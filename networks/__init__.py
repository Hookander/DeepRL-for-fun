from typing import Dict, Type
from networks.base_net import BaseNet
from networks.dqn_small import DQNSmall
from networks.dqn_medium import DQNMedium

model_name_to_ModelClass : Dict[str, Type[BaseNet]] = {
    "small" : DQNSmall,
    "medium" : DQNMedium
}