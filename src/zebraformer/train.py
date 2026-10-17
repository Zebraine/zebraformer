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


if __name__ == "__main__":
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
    x_batched = x.unsqueeze(0)
    full = make_sequences(torch.cat([x_batched, x_batched], dim=0))
    print("x    :", x.tolist())
    print("full :", full.tolist())
    print("shape:", tuple(full.shape))
