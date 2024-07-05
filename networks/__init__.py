from typing import Dict, Type
from networks.base_net import BaseNet
from networks.linear import LinearNetwork
from networks.cnn import CNN
from networks.split_cnn import SplitCNN


network_name_to_ModelClass : Dict[str, Type[BaseNet]] = {
    "linear" : LinearNetwork,
    "cnn" : CNN,
    "split_cnn" : SplitCNN
}