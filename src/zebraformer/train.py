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


def answer_loss(logits: torch.Tensor, targets: torch.Tensor, n: int) -> torch.Tensor:
    logits = logits[:, n : 2 * n + 1]
    targets = targets[:, n : 2 * n + 1]
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))


@torch.no_grad()
def evaluate(model: nn.Module, n: int, num_samples: int = 256):
    model.eval()

    g = torch.Generator().manual_seed(1234)
    x = torch.randint(0, DIGITS, (num_samples, n), generator=g)
    full = make_sequences(x)

    logits: torch.Tensor = model(full[:, :-1])
    pred = logits[:, n : 2 * n + 1].argmax(dim=-1)
    target = full[:, 1:][:, n : 2 * n + 1]

    token_acc = (pred == target).float().mean().item()
    exact = (pred == target).all(dim=1).float().mean().item()
    per_pos = (pred == target).float().mean(dim=0)
    return token_acc, exact, per_pos


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
    torch.manual_seed(0)

    n = 8
    batch_size = 64
    steps = 500

    g = torch.Generator().manual_seed(0)
    model = Bigram()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for step in range(1, steps + 1):
        xb, yb = make_batch(batch_size, n, g)

        logits = model(xb)
        loss = answer_loss(logits, yb, n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step}: loss {loss.item():.4f}")

    token_acc, exact, per_pos = evaluate(model, n)
    print(f"token acc: {token_acc:.4f}, exact: {exact:.4f}")
    print(f"per pos: {per_pos}")
