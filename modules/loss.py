import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

class Load_Balancing_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, f, P,):
        return torch.dot(f, P)