import math

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


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        self.out_proj = nn.Linear(d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)

        T = x.size(1)
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attn_output = weights @ v

        return self.out_proj(attn_output)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_heads = n_head
        self.d_head = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)

        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        attn_output = weights @ v

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)


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
