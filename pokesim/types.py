import torch
import numpy as np

from typing import Dict, Any

TensorType = np.ndarray | torch.Tensor
TensorDict = Dict[str, TensorType | Any]
