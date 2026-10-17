import torch
import torch.nn.functional as F
from torch import nn

DIGITS = 10
BOS = 10
EOS = 11
VOCAB_SIZE = 12


def make_sequences(x: torch.Tensor) -> torch.Tensor:
    batch_size = x.size(0)
    bos = torch.full((batch_size, 1), BOS, dtype=torch.long)
    eos = torch.full((batch_size, 1), EOS, dtype=torch.long)
    return torch.cat([bos, x, x.flip(1), eos], dim=1)


def make_batch(batch_size: int, n: int, generator: torch.Generator):
    x = torch.randint(0, DIGITS, (batch_size, n), generator=generator)
    full = make_sequences(x)
    return full[:, :-1], full[:, 1:]

class Bigram(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        logits = self.head(x)
        return logits

if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    xb, yb = make_batch(2, 8, g)

    model = Bigram()
    logits: torch.Tensor = model(xb)

    print("logits shape:", logits.shape)
    print("loss:", F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), yb.reshape(-1)).item())
    print("params:", sum(p.numel() for p in model.parameters()))
