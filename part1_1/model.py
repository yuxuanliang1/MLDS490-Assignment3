import torch
from torch import nn


class CharacterMLP(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 62))


def initial_model(seed, device="cpu"):
    # Isolate initialization from sampling and mini-batch random streams.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return CharacterMLP().to(device=device, dtype=torch.float32)


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}
