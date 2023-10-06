# Author: Guanxiong


import torch
from typing import Dict


def move_dict_to_device(
    dict: Dict[str, torch.Tensor],
    device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in dict.items()}