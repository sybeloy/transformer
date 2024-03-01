import json
import math
import time

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from dataset import Tokenizer, Dataloader

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cpu'

print('Device:', device)


class FFN(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class AddNorm(nn.Module):
    def __init__(self, emb_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        y = self.dropout(y)
        return self.norm(x + y)


class MHA(nn.Module):
    def __init__(self, emb_dim, n_heads, seq_len):
        super().__init__()
        assert not emb_dim % n_heads
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads

        self.key_proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.query_proj = nn.Linear(emb_dim, emb_dim, bias=True)
        self.value_proj = nn.Linear(emb_dim, emb_dim, bias=True)

        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=True)

        self.register_buffer('tril',
                             torch.tril(torch.ones(seq_len, seq_len)))

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

    def join_heads(self, inputs):
        # inputs (B, H, L, HD) -> (B, L, H, HD) ->  (B, L, D)
        inputs = inputs.permute(0, 2, 1, 3)
        return inputs.reshape(*inputs.shape[:-2], -1)

    def forward(self, inputs):
        k = self.split_heads(self.key_proj(inputs))
        q = self.split_heads(self.query_proj(inputs))
        v = self.split_heads(self.value_proj(inputs))

        weighted = self.attention(k, q, v)
        weighted = self.join_heads(weighted)

        return self.out_proj(weighted)


class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, dropout, n_heads, seq_len):
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


class BigramLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 seq_len: int,
                 batch_size: int,
                 n_layers: int,
                 n_heads: int,
                 dropout: float = 0.1
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoding = nn.Embedding(seq_len, emb_dim)

        self.net = nn.Sequential(*[EncoderLayer(emb_dim, dropout, n_heads, seq_len)
                                   for _ in range(n_layers)])
        self.out_proj = nn.Linear(emb_dim, vocab_size)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.out_proj.bias.data.zero_()
        self.out_proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, tokens, targets=None):
        """
        tokens - input size (B, L)
        """
        _, seq_len = tokens.shape
        token_emb = self.embedding(tokens.long())
        pos_emb = self.pos_encoding(torch.arange(seq_len).to(device))
        token_emb = token_emb + pos_emb

        emb = self.net(token_emb)
        emb = self.out_proj(emb)
        loss = None
        if targets is not None:
            b, l, d = emb.shape
            emb = emb.view(b * l, d)
            targets = targets.view(-1)
            loss = F.cross_entropy(emb, targets.long())
        return emb, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self(idx[:, -self.seq_len:])
            logits = logits[:, -1]
            probs = F.softmax(logits, -1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

    def infinite_generator(self, prompt, tokenizer, memory_offset=None):
        if not memory_offset:
            memory_offset = self.seq_len
        prompt = prompt.unsqueeze(0).to(device)
        while True:
            prompt = prompt[:, -memory_offset:]
            logits, _ = self(prompt)
            logits = logits[:, -1]
            probs = F.softmax(logits, -1)
            next_idx = torch.multinomial(probs, num_samples=1)
            prompt = torch.cat((prompt, next_idx), dim=1)
            yield tokenizer.decode(next_idx)


def train():
    texts = []
    data_paths = ['user1203605613.txt', 'user229875949.txt']
    for data_path in data_paths:
        with open(data_path, 'r') as f:
            texts.append(f.read())
    text = '\n'.join(texts)
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    epochs = 10000

    tkn = Tokenizer(vocab)
    batch_size = 64
    seq_len = 256
    loader = Dataloader(
        batch_size=batch_size,
        seq_len=seq_len,
        tokenizer=tkn,
        text_corpus=text
    )
    model = BigramLM(
        vocab_size=len(vocab),
        emb_dim=240 - 32,
        seq_len=seq_len,
        batch_size=batch_size,
        n_layers=4,
        n_heads=8
    ).to(device)
    # emb, loss = model(x, y)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=epochs
    )
    loss_history = [None]
    pbar = tqdm(range(epochs), total=epochs)
    min_loss = 100
    last_save = 0
    for steps in pbar:
        pbar.set_description(f"Loss {loss_history[-1]}:.3f")

        xb, yb = loader.get_batch('train')
        xb = xb.to(device)
        yb = yb.to(device)

        for param in model.parameters():
            param.grad = None
        with torch.autocast(device_type='cuda', dtype=torch.float16,
                            enabled=torch.cuda.is_available()):
            _, loss = model(xb, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)

        scaler.step(optimizer)

        scaler.update()
        scheduler.step()
        loss_history.append(loss.item())
    torch.save(model.state_dict(), 'model.pth')

    for i in range(5):
        print(i, 'generation')
        x, _ = loader.get_batch('train')
        x = x[0].unsqueeze(0)
        x = x.to(device)
        print(tkn.decode(x[0]))
        print(tkn.decode(model.generate(x, 100)[0]))
    return model


def generate():
    with open('user1203605613.txt', 'r') as f:
        text = f.read()
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)

    tkn = Tokenizer(vocab)
    batch_size = 64
    seq_len = 256
    model = BigramLM(
        vocab_size=len(vocab),
        emb_dim=480,
        seq_len=seq_len,
        batch_size=batch_size,
        n_layers=6,
        n_heads=8
    ).to(device)
    model.load_state_dict(torch.load('model (6).pth', map_location=device))
    model.eval()

    prompt = input('Enter start: ')
    prompt = tkn.encode(prompt)
    for i, word in enumerate(model.infinite_generator(prompt, tkn)):
        print(word, end='')
        if word == '.':
            print()
            time.sleep(2)


if __name__ == '__main__':
    generate()