"""Resumable sequential FedAvg experiments with round-zero and whole-pool metrics."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .federation import Client, aggregate, evaluate_global, select_clients
from .model import clone_state, initial_model
from .utils import (environment, keyed_seed, load_checkpoint, read_json,
                    save_checkpoint, write_csv, write_json)


@dataclass(frozen=True)
class Config:
    rounds: int = 80
    fraction: float = 0.10
    epochs: int = 1
    lr: float = 0.01
    batch_size: int = 32
    seed: int = 2026
    device: str = "cpu"
    split_sha256: str = ""
    data_tag: str = "original_users_inverted_1_minus_x"

    def validate(self):
        if not 0 < self.fraction <= .10:
            raise ValueError("Part 1.1 real experiments require 0 < C <= 0.10")
        if self.rounds < 1 or self.epochs < 1 or self.batch_size < 1 or not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Invalid optimization configuration")


def state_digest(state):
    h = hashlib.sha256()
    for k, v in state.items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_experiment(data, config, output, stop_after=None):
    config.validate()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    expected = asdict(config)
    if (out / "config.json").exists() and read_json(out / "config.json") != expected:
        raise ValueError(f"Cached experiment has different config: {out}")
    write_json(out / "config.json", expected)
    if (out / "summary.json").exists():
        print(f"CACHE {out.name}", flush=True)
        return read_json(out / "summary.json")
    factory = lambda: initial_model(config.seed, config.device)
    clients = {cid: Client(c, factory) for cid, c in data.items()}
    eligible = [cid for cid, client in clients.items() if client.n_train > 0]
    if not eligible:
        raise ValueError("No eligible clients")
    model = factory()
    # Independent sampling streams for each C; the same C shares selections across E.
    rng = np.random.default_rng(keyed_seed(config.seed, "sampling", config.fraction))
    best_key = (-1., -float("inf"))
    best_round = 0
    elapsed_before = 0.
    start_round = 1
    if (out / "last.pt").exists():
        last = load_checkpoint(out / "last.pt")
        model.load_state_dict(last["state"])
        history, selections = last["history"], last["selections"]
        rng.bit_generator.state = last["sampling_rng"]
        best_key, best_round = tuple(last["best_key"]), last["best_round"]
        elapsed_before = last["elapsed_seconds"]
        start_round = last["round"] + 1
        print(f"RESUME {out.name} round {start_round}", flush=True)
    else:
        write_json(out / "environment.json", environment(config.device))
        initial = clone_state(model)
        save_checkpoint(out / "initial.pt", dict(state=initial, config=expected, digest=state_digest(initial)))
        metrics, initial_clients = evaluate_global(model, clients)
        write_csv(out / "client_initial.csv", initial_clients)
        history = [dict(round=0, **metrics, round_seconds=0., cumulative_client_epochs=0,
                        cumulative_examples_seen=0, cumulative_sgd_steps=0)]
        selections = []
        best_key = (metrics["validation_accuracy"], -metrics["validation_loss"])
        save_checkpoint(out / "best.pt", dict(state=initial, round=0, config=expected, metrics=metrics))
    start = time.perf_counter()
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(config.device)
    for round_id in range(start_round, config.rounds + 1):
        round_start = time.perf_counter()
        selected = select_clients(eligible, config.fraction, rng)
        broadcast = clone_state(model)
        updates = []
        for cid in selected:  # Sequential by design; Ray belongs to Part 1.2.
            u = clients[cid].train(broadcast, config.epochs, config.lr, config.batch_size,
                                   keyed_seed(config.seed, "local", round_id, cid))
            updates.append(u)
        new_state = aggregate(updates)  # Parameters + metadata only, no raw examples.
        model.load_state_dict(new_state)
        metrics, _ = evaluate_global(model, clients)
        history.append(dict(round=round_id, **metrics, round_seconds=time.perf_counter() - round_start,
                       cumulative_client_epochs=history[-1]["cumulative_client_epochs"] + len(selected) * config.epochs,
                       cumulative_examples_seen=history[-1]["cumulative_examples_seen"] + sum(u.n_train for u in updates) * config.epochs,
                       cumulative_sgd_steps=history[-1]["cumulative_sgd_steps"] + sum(u.steps for u in updates)))
        selections.extend(dict(round=round_id, selection_order=i, client_id=u.client_id,
                               n_train=u.n_train, epochs=config.epochs, sgd_steps=u.steps)
                          for i, u in enumerate(updates))
        key = (metrics["validation_accuracy"], -metrics["validation_loss"])
        if key > best_key:
            best_key, best_round = key, round_id
            save_checkpoint(out / "best.pt", dict(state=clone_state(model), round=round_id,
                            config=expected, metrics=metrics))
        elapsed = elapsed_before + time.perf_counter() - start
        save_checkpoint(out / "last.pt", dict(state=clone_state(model), round=round_id,
                        config=expected, history=history, selections=selections,
                        sampling_rng=rng.bit_generator.state, best_key=best_key,
                        best_round=best_round, elapsed_seconds=elapsed))
        write_csv(out / "history.csv", history)
        write_csv(out / "selections.csv", selections)
        if round_id == 1 or round_id % 10 == 0 or round_id == config.rounds:
            print(f"{out.name} r={round_id}/{config.rounds} train={metrics['train_accuracy']:.4f} "
                  f"val={metrics['validation_accuracy']:.4f} loss={metrics['validation_loss']:.4f} "
                  f"seconds={elapsed:.1f}", flush=True)
        if stop_after is not None and round_id >= stop_after and round_id < config.rounds:
            return dict(interrupted_after=round_id)
    # Use final model for benefit diagnostic. The separate best checkpoint selects test model.
    _, final_clients = evaluate_global(model, clients)
    write_csv(out / "client_final.csv", final_clients)
    elapsed = elapsed_before + time.perf_counter() - start
    summary = dict(**expected, eligible_clients=len(eligible),
                   clients_per_round=len(select_clients(eligible, config.fraction, np.random.default_rng(0))),
                   initial_digest=load_checkpoint(out / "initial.pt")["digest"],
                   final_train_accuracy=history[-1]["train_accuracy"],
                   final_validation_accuracy=history[-1]["validation_accuracy"],
                   final_validation_loss=history[-1]["validation_loss"],
                   best_validation_accuracy=best_key[0], best_round=best_round,
                   elapsed_seconds=elapsed,
                   peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(config.device) if config.device.startswith("cuda") else 0,
                   **{k: history[-1][k] for k in ("cumulative_client_epochs", "cumulative_examples_seen", "cumulative_sgd_steps")})
    write_json(out / "summary.json", summary)
    return summary
