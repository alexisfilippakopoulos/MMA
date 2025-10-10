# filepath: /MMA/modules/loss.py

"""
loss.py

This module implements custom loss functions for the Mixture of Multimodal Adapters (MMA) architecture, which is designed for sentiment analysis using multimodal inputs. The primary focus is on balancing the contributions of different modalities during training to enhance model performance.

Classes:
    Load_Balancing_loss: A custom loss function that computes the load balancing loss across different modalities. This loss function is crucial for ensuring that the model learns effectively from all available modalities (text, vision, audio) without being biased towards any single modality.

Usage:
    The Load_Balancing_loss class can be instantiated and used during the training process of the MMA model. It takes the outputs from the model and the corresponding modality weights to compute the loss, which is then used to update the model parameters.

Example:
    loss_fn = Load_Balancing_loss()
    loss = loss_fn(outputs, modality_weights)

This implementation is aligned with the MMA architecture as described in the associated research paper, which emphasizes the importance of load balancing in multimodal learning.
"""

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
    """import math
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

    Load_Balancing_loss

    This class implements a load balancing loss function that encourages the model to distribute its learning across different modalities. 
    The loss is computed as the dot product of the feature representations and the modality weights, promoting balanced learning.

    Methods:
        forward(f, P): Computes the load balancing loss.

    Args:
        f (torch.Tensor): The feature representations from the model.
        P (torch.Tensor): The modality weights indicating the importance of each modality.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, f, P):
        """
        forward

        Computes the load balancing loss.

        Args:
            f (torch.Tensor): The feature representations from the model.
            P (torch.Tensor): The modality weights indicating the importance of each modality.

        Returns:
            torch.Tensor: The computed load balancing loss.
        """
        return torch.dot(f, P)