import math

import torch
import torch.nn.functional as F
from torch import nn


class MHA(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            n_heads: int,
            seq_len: int
    ):
        super().__init__()
        assert not emb_dim % n_heads, 'not valid emb_dim and n_heads'
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads

        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.value_proj = nn.Linear(emb_dim, emb_dim, bias=True)

        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=True)

        self.register_buffer('tril',
                             torch.tril(torch.ones(seq_len, seq_len)))

    # TODO: add outer mask
    def attention(self, k, q, v):
        attn = q.matmul(k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
        seq_len = k.shape[-2]
        attn = attn.masked_fill(self.tril[:seq_len, :seq_len] == 0,
                                float('-inf'))
        attn = F.softmax(attn, -1)
        return attn.matmul(v)

    def split_heads(self, inputs):
        # inputs (B, L, D) -> (B, L, H, HD) -> (B, H, L, HD)
        inputs = inputs.reshape(*inputs.shape[:-1], self.n_heads, self.head_dim)
        return inputs.permute(0, 2, 1, 3)

    @staticmethod
    def join_heads(inputs):
        # inputs (B, H, L, HD) -> (B, L, H, HD) ->  (B, L, D)
        inputs = inputs.permute(0, 2, 1, 3)
        return inputs.reshape(*inputs.shape[:-2], -1)

    def forward(self, q, k, v):
        k = self.split_heads(self.key_proj(k))
        q = self.split_heads(self.query_proj(q))
        v = self.split_heads(self.value_proj(v))

        weighted = self.attention(k, q, v)
        weighted = self.join_heads(weighted)

        return self.out_proj(weighted)
