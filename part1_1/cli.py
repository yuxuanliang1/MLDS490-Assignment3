import argparse
from pathlib import Path

from .data import analyze, load_development
from .runner import Config, run_experiment
from .utils import resolve_device, setup, sha256


def main():
    p = argparse.ArgumentParser(description="Assignment 3 Part 1.1 sequential FedAvg")
    p.add_argument("command", choices=["inspect", "train", "pilot", "suite", "validate", "select", "test", "plots", "report"])
    p.add_argument("--data-dir", default="Assignment3-data")
    p.add_argument("--output-dir", default="part1_1_deepdish5_outputs")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--fraction", type=float, default=.1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=490)
    p.add_argument("--name", default="custom")
    args = p.parse_args()
    setup(args.seed, args.threads)
    args.device = resolve_device(args.device)
    root = Path(args.output_dir)
    if args.command in ["plots", "report", "select", "test"]:
        from . import workflow
        return getattr(workflow, args.command)(args)
    data, manifest = load_development(args.data_dir, root/"splits.json", args.device, args.split_seed)
    if args.command == "inspect":
        from .utils import environment, write_json
        print(analyze(data, root/"analysis", manifest))
        write_json(root/"environment.json", environment(args.device))
    elif args.command == "train":
        c = Config(args.rounds, args.fraction, args.epochs, args.lr, args.batch_size,
                   args.seed, args.device, sha256(root/"splits.json"))
        print(run_experiment(data, c, root/"experiments"/args.name))
    else:
        from . import workflow
        getattr(workflow, args.command)(args, data, manifest)
