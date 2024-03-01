import json
import torch
import tiktoken


def form_vocab(text: str) -> list:
    return sorted(set(text))


def save_vocab(data_paths: list):
    texts = []
    for data_path in data_paths:
        with open(data_path, 'r') as f:
            texts.append(f.read())
    text = '\n'.join(texts)
    vocab = form_vocab(text)
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)
    return text, vocab


class SymbolTokenizer:
    def __init__(self, vocab: list):
        self.vocab = vocab
        self.token_to_id = {token: i
                            for i, token in enumerate(vocab)}
        self.id_to_token = {i: token
                            for i, token in enumerate(vocab)}
        self.vocab_size = len(vocab)


    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.token_to_id[token]
                             for token in text], dtype=torch.float)

    def decode(self, token_ids: torch.Tensor) -> str:
        return ''.join(self.id_to_token[id.item()] for id in token_ids)


class Tiktoken:
    def __init__(self, *args):
        self.tkn = tiktoken.encoding_for_model("gpt-4")
        self.vocab_size = self.tkn.max_token_value

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tkn.encode(text), dtype=torch.float)

    def decode(self, token_ids: torch.Tensor) -> str:
        return self.tkn.decode(token_ids)


class Dataloader:
    def __init__(
            self,
            batch_size: int,
            seq_len: int,
            tokenizer: Tiktoken,
            text_corpus: str,
            train_size: float = 0.9
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.data = tokenizer.encode(text_corpus)
        self.split = {
            'train': self.data[:int(train_size * len(self.data))],
            'val': self.data[int(train_size * len(self.data)):]
        }

    def get_batch(self, split: str):
        data = self.split[split]
        idx = torch.randint(0, len(data) - self.seq_len,
                            (self.batch_size,))
        x = torch.stack([data[i: i + self.seq_len]
                         for i in idx])
        y = torch.stack([data[i + 1: i + self.seq_len + 1]
                         for i in idx])
        return x, y


if __name__ == '__main__':
    text, vocab = save_vocab(['user1203605613.txt', 'user229875949.txt'])
    tkn = Tokenizer(vocab)
    loader = Dataloader(
        batch_size=4,
        seq_len=100,
        tokenizer=tkn,
        text_corpus=text
    )
    x, y = loader.get_batch('train')
    print(x.shape)
    print(y.shape)
    print(tkn.decode(x[0]))
