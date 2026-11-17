"""Audit actual saved main runs before test access, then audit final artifacts."""
import argparse
from collections import Counter
import csv
import math
from pathlib import Path

import numpy as np

from .utils import load_checkpoint, read_json, sha256, write_json


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    root = Path("part1_1_deepdish5_outputs")
    split = read_json(root/"splits.json")
    assert sha256("Assignment3-data/train_data.npy") == split["source_sha256"]
    counts = {int(cid): len(v["train"]) for cid, v in split["clients"].items()}
    for v in split["clients"].values():
        tr, va = v["train"], v["validation"]
        assert not set(tr) & set(va)
        assert sorted(tr+va) == list(range(len(tr)+len(va)))
    comparison = read_json(root/"comparison.json")
    audited = []
    for run in comparison:
        path = root/"experiments"/run["name"]
        hist, selection = rows(path/"history.csv"), rows(path/"selections.csv")
        assert [int(r["round"]) for r in hist] == list(range(81))
        assert all(int(r["train_n"]) == 25418 and int(r["validation_n"]) == 6407 for r in hist)
        assert all(math.isfinite(float(r[k])) for r in hist for k in ["train_loss","validation_loss","train_accuracy","validation_accuracy"])
        assert all(0 <= float(r[k]) <= 1 for r in hist for k in ["train_accuracy","validation_accuracy"])
        k = run["clients_per_round"]
        assert k in [2,5,10] and run["fraction"] <= .1
        assert len(selection) == 80*k
        for round_id in range(1,81):
            selected = [r for r in selection if int(r["round"]) == round_id]
            assert len(selected) == k and len({r["client_id"] for r in selected}) == k
            assert all(int(r["n_train"]) == counts[int(r["client_id"])] for r in selected)
        expected_exposures = sum(int(r["n_train"])*int(r["epochs"]) for r in selection)
        assert expected_exposures == run["cumulative_examples_seen"]
        assert run["cumulative_client_epochs"] == 80*k*run["epochs"]
        checkpoint = load_checkpoint(path/"best.pt")
        expected_best = max(hist, key=lambda r:(float(r["validation_accuracy"]),-float(r["validation_loss"]),-int(r["round"])))
        assert checkpoint["round"] == int(expected_best["round"])
        env = read_json(path/"environment.json")
        assert env["hostname"] == "mlds-deepdish5"
        assert env["gpu"] == "NVIDIA GeForce RTX 5060 Ti"
        audited.append(dict(name=run["name"], rounds=80, selections=len(selection),
                            best_round=checkpoint["round"], examples_seen=expected_exposures))
    for seed in [2026,2027,2028]:
        assert len({r["initial_digest"] for r in comparison if r["seed"] == seed}) == 1
    science = read_json(root/"validation/results.json")
    assert all(r["passed"] for r in science)
    assert read_json(root/"validation/correctness_tests.json")["successful"]
    result = dict(passed=True, audited_main_and_repeat_runs=len(audited),
                  all_server_host_and_gpu_verified=True, split_counts_verified=True,
                  all_rounds_and_full_pool_counts_verified=True, all_selection_weights_verified=True,
                  all_best_checkpoints_match_validation_criteria=True, runs=audited)
    if args.final:
        selected, test = read_json(root/"selection.json"), read_json(root/"final_test.json")
        assert sha256(root/"selection.json") == test["selection_sha256"]
        assert sha256(selected["checkpoint"]) == test["checkpoint_sha256"]
        assert sha256("Assignment3-data/test_data.npy") == test["test_source_sha256"]
        assert test["accuracy"] == test["correct"]/test["n"]
        manifest = read_json(root/"figures/manifest.json")
        for f in manifest:
            for ext in ["png","pdf"]:
                assert (root/"figures"/(f["name"]+"."+ext)).stat().st_size > 1000
        assert (root/"REPORT_NOTES.md").stat().st_size > 10000
        result.update(final_selection_and_test_hash_verified=True, figure_pairs=len(manifest))
    write_json(root/("final_audit.json" if args.final else "validation/artifact_audit.json"), result)
    print("ARTIFACT AUDIT PASSED", result, flush=True)


if __name__ == "__main__":
    main()
