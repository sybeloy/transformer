import json
import time

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from encoder import TransformerEncoder
from decoder import TransformerDecoder
from embeddings import TokenEmbedding
from dataset import SymbolTokenizer, Tiktoken, Dataloader

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cpu'

print('Device:', device)


class Transformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 seq_len: int,
                 batch_size: int,
                 n_layers: int,
                 n_heads: int,
                 device: torch.device,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size

        self.encoder = TransformerEncoder(
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            seq_len=seq_len,
            device=device,
            vocab_size=vocab_size
        )
        self.decoder = TransformerDecoder(
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device
        )

    def forward(self, tokens, targets=None):
        """
        tokens - input size (B, L)
        """
        enc_out = self.encoder(tokens)
        output = self.decoder(targets, encoder_inputs=enc_out)
        loss = None
        if targets is not None:
            b, l, d = output.shape
            emb = output.view(b * l, d)
            targets = targets.view(-1)
            loss = F.cross_entropy(emb, targets.long())
        return output, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self.forward(idx[:, -self.seq_len:])
            logits = logits[:, -1]
            probs = F.softmax(logits, -1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

    def infinite_generator(self, prompt, tokenizer, memory_offset=None):
        memory_offset = min(self.seq_len, memory_offset) if memory_offset else self.seq_len
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
    epochs = 500

    tkn = SymbolTokenizer(vocab)
    batch_size = 64
    seq_len = 256
    loader = Dataloader(
        batch_size=batch_size,
        seq_len=seq_len,
        tokenizer=tkn,
        text_corpus=text
    )
    model = Transformer(
        vocab_size=tkn.vocab_size,
        emb_dim=320,
        seq_len=seq_len,
        batch_size=batch_size,
        n_layers=3,
        n_heads=8,
        device=device
    ).to(device)
    # emb, loss = model(x, y)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=epochs
    )
    loss_history = [10]
    pbar = tqdm(range(epochs), total=epochs)
    min_loss = 100
    last_save = 0
    for steps in pbar:
        pbar.set_description(f"Loss {loss_history[-1]:.3f}")

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
    model = Transformer(
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
    train()
