# filepath: /MMA/MMA/modules/encoders.py

"""
This module defines the encoder components used in the MMA (Mixture of Multimodal Adapters) architecture for sentiment analysis. 
The MMA architecture integrates multiple modalities, including text, vision, and audio, to enhance sentiment classification tasks.

Classes:
- SubNet: A subnetwork designed for processing audio and video inputs in the pre-fusion stage. It consists of fully connected layers with dropout for regularization.
- RouterSelfAttention: Implements self-attention mechanisms that allow the model to focus on different parts of the input sequence, enhancing the representation of multimodal data.
- RouterPFSelfAttention: A variant of self-attention that processes inputs without additional linear projections, optimizing the attention mechanism for efficiency.
- RouterPFMultiHeadAttention: A multi-head attention mechanism that allows the model to attend to different parts of the input across multiple heads, facilitating better representation learning.

The encoder components are crucial for extracting meaningful features from the multimodal inputs, which are then fused and processed by the MMA architecture for sentiment analysis.
"""

import torch
import torch.nn.functional as F
import time
import math
from torch import nn

class SubNet(nn.Module):
    '''
    The subnetwork that is used in the MMA architecture for processing audio and video inputs in the pre-fusion stage.

    Args:
        in_size (int): Input dimension.
        hidden_size (int): Hidden layer dimension.
        n_class (int): Number of output classes.
        dropout (float): Dropout probability for regularization.

    Output:
        A tensor of shape (batch_size, hidden_size) representing the processed features.
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout):
        super(SubNet, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Forward pass for the SubNet.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, n_class) representing the class scores.
        '''
        dropped = self.drop(x)
        y_1 = torch.relu(self.linear_1(dropped))
        fusion = self.linear_2(y_1)
        y_2 = torch.relu(fusion)
        y_3 = self.linear_3(y_2)
        return y_3

class RouterSelfAttention(nn.Module):
    '''
    Implements self-attention mechanism for the MMA architecture.

    Args:
        embed_dim (int): Dimension of the input embeddings.
        attn_dim (int): Dimension of the attention mechanism.

    Output:
        A tensor representing the attended features.
    '''

    def __init__(self, embed_dim=768, attn_dim=2):
        super(RouterSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim if attn_dim else embed_dim
        
        self.query = nn.Linear(embed_dim, self.attn_dim)
        self.key = nn.Linear(embed_dim, self.attn_dim)
        self.value = nn.Linear(embed_dim, self.attn_dim)
        self.out = nn.Linear(self.attn_dim, self.attn_dim)
        
    def forward(self, x):
        '''
        Forward pass for the RouterSelfAttention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, attn_dim) representing the attended features.
        '''
        batch_size, seq_length, embed_dim = x.size()
        assert embed_dim == self.embed_dim, "Embedding dimension must match"
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attn_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        output = self.out(attn_output)
        
        return output
    
class RouterPFSelfAttention(nn.Module):
    '''
    Implements a variant of self-attention for the MMA architecture without additional linear projections.

    Args:
        embed_dim (int): Dimension of the input embeddings.
        attn_dim (int): Dimension of the attention mechanism.

    Output:
        A tensor representing the attended features.
    '''

    def __init__(self, embed_dim=768, attn_dim=2):
        super(RouterPFSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim if attn_dim else embed_dim
        
        self.out = nn.Linear(self.embed_dim, self.attn_dim)
        
    def forward(self, x):
        '''
        Forward pass for the RouterPFSelfAttention.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, attn_dim) representing the attended features.
        '''
        batch_size, seq_length, embed_dim = x.size()
        assert embed_dim == self.embed_dim, "Embedding dimension must match"
        
        Q = x
        K = x
        V = x
        
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        output = self.out(attn_output)
        
        return output

class RouterPFMultiHeadAttention(nn.Module):
    '''
    Implements multi-head attention mechanism for the MMA architecture.

    Args:
        num_heads (int): Number of attention heads.
        embed_size (int): Dimension of the input embeddings.

    Output:
        A tensor representing the attended features across multiple heads.
    '''

    def __init__(self, num_heads=12, embed_size=768):
        super(RouterPFMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

    def forward(self, queries, mask=None):
        '''
        Forward pass for the RouterPFMultiHeadAttention.

        Args:
            queries (Tensor): Input tensor of shape (batch_size, query_len, embed_size).
            mask (Tensor, optional): Mask tensor for attention scores.

        Returns:
            Tensor: Output tensor of shape (batch_size, query_len, embed_size) representing the attended features.
        '''
        N = queries.shape[0]
        values = queries
        keys = queries
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention_scores = attention_scores / (self.embed_size ** (1/2))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(attention_scores, dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        out = self.out(out)

        return out