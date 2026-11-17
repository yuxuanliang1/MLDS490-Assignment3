"""Ordered experiment workflow; test access is isolated behind frozen selection."""
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from .data import analyze, validate_arrays
from .federation import evaluate_tensors
from .model import initial_model
from .runner import Config, run_experiment
from .utils import load_checkpoint, read_json, sha256, write_csv, write_json


def pilot(args, data, manifest):
    root = Path(args.output_dir)
    analyze(data, root/"analysis", manifest)
    rows = []
    for lr in [.01, .1, .3]:
        for epochs in [1, 3]:
            name = f"pilot_ink_lr{lr}_e{epochs}"
            cfg = Config(12, .1, epochs, lr, 32, 2026, args.device, sha256(root/"splits.json"))
            s = run_experiment(data, cfg, root/"experiments"/name)
            rows.append(dict(name=name, **s))
    # Shared learning rate selected using mean FINAL validation accuracy across E=1,3.
    means = {lr: float(np.mean([r["final_validation_accuracy"] for r in rows if r["lr"] == lr]))
             for lr in [.01, .1, .3]}
    lr = max(means, key=lambda x: (means[x], -x))
    result = dict(lr=lr, rounds=80, batch_size=32,
                  selection_rule="Highest mean round-12 validation accuracy across E=1,3; smaller LR breaks tie",
                  rationale="Full-pool pilot improves with selected LR; 80 rounds retained as common predeclared budget. Test unopened.",
                  mean_validation_by_lr=means,
                  estimated_grid_seconds=float(sum(r["elapsed_seconds"] for r in rows) / len(rows) * (80/12) * 6),
                  pilots=rows)
    write_json(root/"pilot_decision.json", result)
    print(result, flush=True)


def suite(args, data, manifest):
    root = Path(args.output_dir)
    decision = read_json(root/"pilot_decision.json")
    rows = []
    for fraction in [.02, .05, .10]:
        for epochs in [1, 3]:
            name = experiment_name(fraction, epochs, 2026)
            cfg = Config(decision["rounds"], fraction, epochs, decision["lr"], decision["batch_size"],
                         2026, args.device, sha256(root/"splits.json"))
            rows.append(dict(name=name, **run_experiment(data, cfg, root/"experiments"/name)))
            write_csv(root/"comparison.csv", rows)
    # Freeze the top two settings before extra seeds, using only primary-seed validation.
    primary = sorted(rows, key=lambda r: (-r["best_validation_accuracy"], -r["final_validation_accuracy"],
                                           r["cumulative_client_epochs"]))[:2]
    write_json(root/"repeat_candidates.json", primary)
    for r in primary:
        for seed in [2027, 2028]:
            name = experiment_name(r["fraction"], r["epochs"], seed)
            cfg = Config(decision["rounds"], r["fraction"], r["epochs"], decision["lr"], decision["batch_size"],
                         seed, args.device, sha256(root/"splits.json"))
            rows.append(dict(name=name, **run_experiment(data, cfg, root/"experiments"/name)))
            write_csv(root/"comparison.csv", rows)
    write_json(root/"comparison.json", rows)


def experiment_name(fraction, epochs, seed):
    return f"c{int(fraction*100):02d}_e{epochs}_s{seed}"


def validate(args, data, manifest):
    from .science import run_validations
    run_validations(args, data, manifest)


def select(args):
    root = Path(args.output_dir)
    if (root/"selection.json").exists():
        print("Frozen selection already exists", read_json(root/"selection.json"))
        return
    results = read_json(root/"validation/results.json")
    if not all(r["passed"] for r in results if r.get("gating", False)):
        raise RuntimeError("Correctness validation failed; do not select/evaluate test")
    rows = read_json(root/"comparison.json")
    repeated = {}
    for r in rows:
        key = (r["fraction"], r["epochs"])
        repeated.setdefault(key, []).append(r)
    stats = []
    for (c, e), rs in repeated.items():
        if len(rs) < 3:
            continue
        vals = [r["best_validation_accuracy"] for r in rs]
        stats.append(dict(fraction=c, epochs=e, n_seeds=len(rs),
                          mean_best_validation=float(np.mean(vals)),
                          sd_best_validation=float(np.std(vals, ddof=1)),
                          mean_final_validation=float(np.mean([r["final_validation_accuracy"] for r in rs])),
                          sd_final_validation=float(np.std([r["final_validation_accuracy"] for r in rs], ddof=1))))
    best = max(stats, key=lambda r: (r["mean_best_validation"], r["mean_final_validation"], -r["epochs"]))
    candidates = repeated[(best["fraction"], best["epochs"])]
    chosen = max(candidates, key=lambda r: (r["best_validation_accuracy"], r["final_validation_accuracy"], -r["seed"]))
    checkpoint = root/"experiments"/chosen["name"]/"best.pt"
    selection = dict(rule="Among top two primary-seed settings, maximize mean best validation over seeds 2026,2027,2028; then choose seed/checkpoint with highest validation accuracy (loss breaks checkpoint ties)",
                     configuration=chosen, repeated_summary=stats,
                     checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                     selected_round=load_checkpoint(checkpoint)["round"],
                     test_previously_opened=False,
                     frozen_at_unix_seconds=time.time())
    write_json(root/"selection.json", selection)
    write_csv(root/"repeat_summary.csv", stats)
    print(selection, flush=True)


def test(args):
    """The ONLY function opening held-out data. Cached result prevents repeated scoring."""
    root = Path(args.output_dir)
    selection = read_json(root/"selection.json")
    selection_hash = sha256(root/"selection.json")
    if (root/"final_test.json").exists():
        result = read_json(root/"final_test.json")
        if result["selection_sha256"] != selection_hash:
            raise ValueError("Selection changed after test evaluation")
        print("CACHE held-out evaluation", result)
        return
    checkpoint = Path(selection["checkpoint"])
    if sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    payload = load_checkpoint(checkpoint)
    model = initial_model(payload["config"]["seed"], args.device)
    model.load_state_dict(payload["state"])
    path = Path(args.data_dir)/"test_data.npy"
    raw = np.load(path, allow_pickle=True)
    if raw.shape != (1,) or set(raw[0]) != {"images", "labels"}:
        raise ValueError("Expected one aggregate held-out dictionary")
    images, labels = validate_arrays(raw[0]["images"], raw[0]["labels"])
    metrics = evaluate_tensors(model, torch.as_tensor(1.0 - images, device=args.device),
                              torch.as_tensor(labels, device=args.device))
    result = dict(accuracy=metrics["correct"] / metrics["n"], correct=metrics["correct"],
                  n=metrics["n"], loss=metrics["loss_sum"] / metrics["n"],
                  test_source_sha256=sha256(path), selection_sha256=selection_hash,
                  checkpoint_sha256=selection["checkpoint_sha256"], selected_round=payload["round"],
                  test_container_shape=list(raw.shape), image_shape=list(images.shape),
                  image_dtype=str(images.dtype), label_dtype=str(np.asarray(raw[0]["labels"]).dtype),
                  note="Single aggregate partition, not 100 per-user test sets. Opened only after selection frozen.")
    write_json(root/"final_test.json", result)
    print(result, flush=True)


def plots(args):
    from .plotting import generate
    generate(Path(args.output_dir))


def report(args):
    from .documentation import generate
    generate(Path(args.output_dir))
