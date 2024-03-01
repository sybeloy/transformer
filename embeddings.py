import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_seq_len, device):
        super().__init__()
        self.encoding = torch.zeros(max_seq_len, emb_dim, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_seq_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, emb_dim, step=2, device=device).float()

        base = torch.tensor(10000, dtype=torch.int32)
        self.encoding[:, 0::2] = torch.sin(pos / (base ** (_2i / emb_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (base ** (_2i / emb_dim)))

    def forward(self, x):
        _, seq_len = x.size()
        return self.encoding[:seq_len, :]


class LearnablePositionalEncoding(nn.Module):
    def __init__(
            self,
            emb_dim: int,
            max_seq_len: int,
            device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.pos_embedding = nn.Embedding(max_seq_len, emb_dim)

    def forward(self, inputs):
        _, seq_len = inputs.shape
        positions = torch.arange(seq_len).to(self.device)
        return self.pos_embedding(positions)


class TokenEmbedding(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            emb_dim: int,
            max_seq_len: int,
            device: torch.device,
            pos_encoding: nn.Module = PositionalEncoding
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = pos_encoding(
            emb_dim=emb_dim,
            max_seq_len=max_seq_len,
            device=device
        )

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        pos_emb = self.pos_embedding(inputs)

        return embeddings + pos_emb
