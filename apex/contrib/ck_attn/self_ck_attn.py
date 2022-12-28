import math

import torch
from torch import nn
from torch.nn import Parameter
#import torch.nn.functional as F

from .self_ck_attn_func import self_attn_func

class SelfCKAttn(nn.Module):

    def __init__(
        self,
        embed_dim,   # config.hidden_size
        num_heads,   # config.num_attention_heads
        dropout=0.0, # config.attention_probs_dropout_prob
        bias=False,
        separate_qkv_params=False,
        best_op_id = 0,
        #num_blocks = 16, 
        #block_size_k = 64, 
        #block_size_o = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads #self.attention_head_size
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.bias = bias
        self.scaling = self.head_dim ** -0.5
        self.separate_qkv_params = separate_qkv_params

        if separate_qkv_params:
            self.q_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.v_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        else:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if self.bias:
            if separate_qkv_params:
                self.q_bias = Parameter(torch.Tensor(embed_dim))
                self.k_bias = Parameter(torch.Tensor(embed_dim))
                self.v_bias = Parameter(torch.Tensor(embed_dim))
            else:
                self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
            self.out_proj_bias = Parameter(torch.Tensor(embed_dim))
        else:
            if separate_qkv_params:
                self.register_parameter("q_bias", None)
                self.register_parameter("k_bias", None)
                self.register_parameter("v_bias", None)
                self.q_bias = None
                self.k_bias = None
                self.v_bias = None
            else:
                self.register_parameter("in_proj_bias", None)
                self.in_proj_bias = None
            self.register_parameter("out_proj_bias", None)
            self.out_proj_bias = None

        self.reset_parameters()
        self.attn_func = self_attn_func

        self.best_op_id = best_op_id

    def reset_parameters(self):
        if self.separate_qkv_params:
            nn.init.xavier_uniform_(self.q_weight)
            nn.init.xavier_uniform_(self.k_weight)
            nn.init.xavier_uniform_(self.v_weight)
        else:
            # in_proj_weight has shape [3 * hidden, hidden] but it should be
            # initialized like a [hidden, hidden] matrix.
            # sqrt(6 / (hidden + hidden)) / sqrt(6 / (3 * hidden + hidden)) = sqrt(2)
            # therefore xavier_uniform gain should be set to sqrt(2).
            nn.init.xavier_uniform_(self.in_proj_weight, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj_weight)
        if self.bias:
            if self.separate_qkv_params:
                nn.init.constant_(self.q_bias, 0.0)
                nn.init.constant_(self.k_bias, 0.0)
                nn.init.constant_(self.v_bias, 0.0)
            else:
                nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj_bias, 0.0)

    def forward(self, query, key, value):

        if self.separate_qkv_params:
            input_weights = (
                torch.cat(
                    [
                        self.q_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim),
                        self.k_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim),
                        self.v_weight.view(self.num_heads, 1, self.head_dim, self.embed_dim),
                    ],
                    dim=1,
                )
                .reshape(3 * self.embed_dim, self.embed_dim)
                .contiguous()
            )
        else:
            input_weights = self.in_proj_weight

        if self.bias:
            if self.separate_qkv_params:
                input_bias = (
                    torch.cat(
                        [
                            self.q_bias.view(self.num_heads, 1, self.head_dim),
                            self.k_bias.view(self.num_heads, 1, self.head_dim),
                            self.v_bias.view(self.num_heads, 1, self.head_dim),
                        ],
                        dim=1,
                    )
                    .reshape(3 * self.embed_dim)
                    .contiguous()
                )
            else:
                input_bias = self.in_proj_bias
        else:
            input_bias = None

        outputs = self.attn_func(
            self.num_heads,
            query,
            input_weights,
            self.out_proj_weight,
            input_bias,
            self.out_proj_bias,
            self.dropout,
            self.best_op_id,
        )
        return outputs, None

  
