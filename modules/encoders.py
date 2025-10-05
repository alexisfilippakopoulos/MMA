import torch
import torch.nn.functional as F
import time
import math
from torch import nn

class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, n_class, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        # self.norm = nn.LayerNorm(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        
        dropped = self.drop(x)
        # normed = self.norm(dropped)
        y_1 = torch.relu(self.linear_1(dropped))
        fusion = self.linear_2(y_1)
        y_2 = torch.relu(fusion)
        y_3 = (self.linear_3(y_2))
        # return y_3,fusion
        return y_3

import torch
import torch.nn as nn
import torch.nn.functional as F

class RouterSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, attn_dim=2):
        super(RouterSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim if attn_dim else embed_dim
        
        self.query = nn.Linear(embed_dim, self.attn_dim)
        self.key = nn.Linear(embed_dim, self.attn_dim)
        self.value = nn.Linear(embed_dim, self.attn_dim)
        self.out = nn.Linear(self.attn_dim, self.attn_dim)
        
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()  # batch_size, 3, 768
        assert embed_dim == self.embed_dim, "Embedding dimension must match"
        
        # Linear projections
        Q = self.query(x)  # (batch_size, seq_length, attn_dim)
        K = self.key(x)    # (batch_size, seq_length, attn_dim)
        V = self.value(x)  # (batch_size, seq_length, attn_dim)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attn_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Put through final linear layer
        output = self.out(attn_output)
        
        return output
    
class RouterPFSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, attn_dim=2):
        super(RouterPFSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim if attn_dim else embed_dim
        
        # self.query = nn.Linear(embed_dim, self.attn_dim)
        # self.key = nn.Linear(embed_dim, self.attn_dim)
        # self.value = nn.Linear(embed_dim, self.attn_dim)
        self.out = nn.Linear(self.embed_dim, self.attn_dim)
        
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()  # batch_size, 3, 768
        assert embed_dim == self.embed_dim, "Embedding dimension must match"
        
        # Linear projections
        Q =(x)  # (batch_size, seq_length, attn_dim)
        K = (x)    # (batch_size, seq_length, attn_dim)
        V = (x)  # (batch_size, seq_length, attn_dim)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Put through final linear layer
        output = self.out(attn_output)
        # output = attn_output.sum(dim=1)
        
        return output
class RouterPFMultiHeadAttention(nn.Module):
    def __init__(self, num_heads=12, embed_size=768):
        super(RouterPFMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads
        self.out = nn.Linear(in_features=768,  out_features=2)

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by num_heads"

    def forward(self, queries, mask=None):
        N = queries.shape[0]
        values = queries
        keys = queries
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.num_heads different pieces
        # print(values.shape)
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        # Perform scaled dot-product attention on each head
        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention_scores = attention_scores / (self.embed_size ** (1/2))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))
            # print(attention_scores.shape)

        attention = torch.softmax(attention_scores, dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        out = self.out(out)

        return out

