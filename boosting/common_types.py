import os
from typing import Union, Callable, Any

import torch

# generic
PathLike = Union[os.PathLike, str]
Function = Callable[..., Any]

# numbers
Number = Union[int, float]
PositiveNumber = Number

# torch
TorchDevice = Union[torch.device, str]
