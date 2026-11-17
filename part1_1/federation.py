"""Parameter-only server aggregation and private client training/evaluation."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from .model import clone_state


@dataclass
class ClientUpdate:
    client_id: int
    n_train: int
    state: dict
    steps: int = 0


def select_clients(eligible, fraction, rng):
    if not 0 < fraction <= 1 or not eligible or len(set(eligible)) != len(eligible):
        raise ValueError("Need unique eligible clients and fraction in (0,1]")
    k = max(1, int(math.floor(fraction * len(eligible) + 1e-9)))
    return rng.choice(sorted(eligible), size=k, replace=False).tolist()


def aggregate(updates):
    """w_next = sum(n_train_k * w_k) / sum(n_train_k), selected clients only."""
    if not updates or len({u.client_id for u in updates}) != len(updates):
        raise ValueError("No updates or duplicate client IDs")
    if any(not isinstance(u.n_train, int) or u.n_train <= 0 for u in updates):
        raise ValueError("Aggregation requires positive integer local TRAINING sample counts")
    keys = list(updates[0].state)
    if not keys:
        raise ValueError("Empty parameter dictionary")
    total = sum(u.n_train for u in updates)
    result = {}
    for key in keys:
        first = updates[0].state[key]
        if first.dtype != torch.float32:
            raise ValueError("FedAvg requires float32 parameters")
        acc = torch.zeros_like(first)
        for u in updates:
            if list(u.state) != keys:
                raise ValueError("Mismatched parameter keys")
            v = u.state[key]
            if v.shape != first.shape or v.dtype != first.dtype or v.device != first.device:
                raise ValueError(f"Incompatible parameter {key}")
            if not torch.isfinite(v).all():
                raise FloatingPointError(f"Non-finite update from client {u.client_id}: {key}")
            acc.add_(v, alpha=u.n_train / total)
        if not torch.isfinite(acc).all():
            raise FloatingPointError("Non-finite aggregated model")
        result[key] = acc
    return result


class Client:
    """Owns local tensors; server receives only parameters, counts and metric sums."""
    def __init__(self, data, model_factory):
        self.data = data
        self.model_factory = model_factory

    @property
    def client_id(self):
        return self.data.client_id

    @property
    def n_train(self):
        return self.data.n_train

    def train(self, global_state, epochs, lr, batch_size, seed):
        if self.n_train == 0:
            raise ValueError(f"Client {self.client_id} has no training examples and is ineligible")
        if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
            raise ValueError("Need positive epochs, batch size and finite learning rate")
        # New model AND optimizer: each client always starts from the identical broadcast.
        model = self.model_factory()
        model.load_state_dict(global_state)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        criterion = nn.CrossEntropyLoss()
        steps = 0
        for _ in range(epochs):
            permutation = torch.randperm(self.n_train, generator=generator).to(self.data.train_y.device)
            for idx in permutation.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(self.data.train_x[idx]), self.data.train_y[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite local loss, client {self.client_id}")
                loss.backward()
                optimizer.step()
                steps += 1
        state = clone_state(model)
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise FloatingPointError(f"Non-finite final parameters, client {self.client_id}")
        return ClientUpdate(self.client_id, self.n_train, state, steps)

    def evaluate(self, model, partition, batch_size=1024):
        if partition not in ("train", "validation"):
            raise ValueError(partition)
        x, y = ((self.data.train_x, self.data.train_y) if partition == "train"
                else (self.data.val_x, self.data.val_y))
        return evaluate_tensors(model, x, y, batch_size)


@torch.no_grad()
def evaluate_tensors(model, x, y, batch_size=1024):
    model.eval()
    loss_sum = torch.zeros((), device=y.device, dtype=torch.float64)
    correct = torch.zeros((), device=y.device, dtype=torch.int64)
    for start in range(0, len(y), batch_size):
        logits = model(x[start:start + batch_size])
        targets = y[start:start + batch_size]
        # Float32 network and CE, double accumulator for stable sample-weighted reporting.
        loss_sum += nn.functional.cross_entropy(logits, targets, reduction="sum").double()
        correct += (logits.argmax(1) == targets).sum()
    if not torch.isfinite(loss_sum):
        raise FloatingPointError("Non-finite evaluation loss")
    return dict(loss_sum=float(loss_sum), correct=int(correct), n=len(y))


def evaluate_global(model, clients):
    totals, per_client = {}, []
    for partition in ("train", "validation"):
        summaries = []
        for cid, client in clients.items():
            s = client.evaluate(model, partition)
            summaries.append(s)
            if partition == "validation":
                per_client.append(dict(client_id=cid, **s, accuracy=s["correct"] / s["n"] if s["n"] else None))
        n = sum(s["n"] for s in summaries)
        if n == 0:
            raise ValueError(f"No aggregate {partition} examples")
        totals.update({f"{partition}_loss": sum(s["loss_sum"] for s in summaries) / n,
                       f"{partition}_accuracy": sum(s["correct"] for s in summaries) / n,
                       f"{partition}_n": n})
    return totals, per_client
