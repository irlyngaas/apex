import math

import torch
from torch import nn
from torch.nn import Parameter
#import torch.nn.functional as F

from .self_ck_attn_func import self_attn_func

class SelfCKAttn(nn.Module):

    def __init__(
        self,
        num_sequences,
        seq_length,
        hidden_dim,
        num_heads,
        head_dim,
        #embed_dim,   # config.hidden_size
        #num_heads,   # config.num_attention_heads
        dropout=0.0, # config.attention_probs_dropout_prob
        best_op_id = 0,
        #num_blocks = 16, 
        #block_size_k = 64, 
        #block_size_o = 64,
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.embed_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.embed_dim // self.num_heads #self.attention_head_size
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.attn_func = self_attn_func
        self.best_op_id = best_op_id

    def forward(self, query, key, value, out):

        outputs = self.attn_func(
            query,
            key,
            value,
            out,
            self.num_sequences,
            self.seq_length,
            self.embed_dim,
            self.num_heads,
            self.head_dim,
            self.dropout,
            self.best_op_id,
        )
        return outputs, None

  
