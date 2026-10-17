import torch

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


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    xb, yb = make_batch(2, 8, g)

    print("input :", xb.tolist())
    print("target:", yb.tolist())
    print("input shape :", xb.shape)
    print("target shape:", yb.shape)
    print(torch.equal(xb[:, 1:], yb[:, :-1]))