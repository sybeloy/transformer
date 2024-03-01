from torch import nn

from attention import MHA
from block_modules import FFN, AddNorm


class EncoderLayer(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            dropout: float,
            n_heads: int,
            seq_len: int
    ):
        super().__init__()
        self.mha = MHA(emb_dim, n_heads, seq_len)
        self.ffn = FFN(emb_dim)
        self.first_add_norm = AddNorm(emb_dim, dropout)
        self.second_add_norm = AddNorm(emb_dim, dropout)

    def forward(self, inputs):
        weighted = self.mha(inputs)
        residual = self.first_add_norm(inputs, weighted)
        projected = self.ffn(residual)
        return self.second_add_norm(residual, projected)


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            n_layers: int,
            dropout: float,
            n_heads: int,
            seq_len: int
    ):
        super().__init__()
        self.layers = nn.Sequential(*[
            EncoderLayer(
                emb_dim=emb_dim,
                n_heads=n_heads,
                dropout=dropout,
                seq_len=seq_len
            ) for _ in range(n_layers)
        ])

    def forward(self, inputs):
        return self.layers(inputs)
