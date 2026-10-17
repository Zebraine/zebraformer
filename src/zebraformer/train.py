import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.profiler import log

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


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Zebraformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 64,
        n_head: int = 4,
        n_layer: int = 2,
    ):
        super().__init__()

        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
            )

        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits


if __name__ == "__main__":
    torch.manual_seed(0)

    n = 8
    g = torch.Generator().manual_seed(0)
    xb, yb = make_batch(2, n, g)

    model = Zebraformer(VOCAB_SIZE, block_size=2 * n + 2)
    logits = model(xb)

    print("logits shape:", logits.shape)
    print("初始 loss:", round(answer_loss(logits, yb, n).item(), 4))
    print("参数量:", sum(p.numel() for p in model.parameters()))
