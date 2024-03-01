import torch
from torch import nn
from block_modules import FFN, AddNorm
from attention import MHA
from embeddings import TokenEmbedding


class DecoderLayer(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            n_heads: int,
            seq_len: int,
            dropout: float,
    ):
        super().__init__()
        self.input_mha = MHA(
            emb_dim=emb_dim,
            n_heads=n_heads,
            seq_len=seq_len
        )
        self.first_add_norm = AddNorm(emb_dim=emb_dim, dropout=dropout)

        self.encoder_mha = MHA(
            emb_dim=emb_dim,
            n_heads=n_heads,
            seq_len=seq_len
        )
        self.second_add_norm = AddNorm(emb_dim=emb_dim, dropout=dropout)

        self.ffn = FFN(emb_dim=emb_dim)
        self.third_add_norm = AddNorm(emb_dim=emb_dim, dropout=dropout)

    def forward(self, inputs, encoder_inputs=None):
        weighted = self.input_mha(inputs, inputs, inputs)
        inputs = self.first_add_norm(inputs, weighted)

        if encoder_inputs is not None:
            enc_dec_mix = self.encoder_mha(
                q=encoder_inputs,
                k=inputs,
                v=inputs
            )
            weighted = self.second_add_norm(weighted, enc_dec_mix)

        weighted = self.third_add_norm(weighted, self.ffn(weighted))
        return weighted


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            n_heads: int,
            seq_len: int,
            dropout: float,
            n_layers: int,
            vocab_size: int,
            device: torch.device
    ):
        super().__init__()
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            max_seq_len=seq_len,
            device=device
        )
        self.layers = nn.ModuleList(
            DecoderLayer(
                emb_dim=emb_dim,
                n_heads=n_heads,
                seq_len=seq_len,
                dropout=dropout,
            ) for _ in range(n_layers)
        )
        self.out_proj = nn.Linear(emb_dim, vocab_size)

    def forward(self, inputs, encoder_inputs=None):
        inputs = self.embedding(inputs)
        for layer in self.layers:
            inputs = layer(inputs, encoder_inputs)
        return self.out_proj(inputs)
