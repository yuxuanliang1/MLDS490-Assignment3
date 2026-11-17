"""Scientific controls A-G. All controls use development data or synthetic data."""
from dataclasses import replace
from pathlib import Path
import csv

import numpy as np
import torch

from .data import ClientData, label_statistics
from .federation import select_clients
from .runner import Config, run_experiment
from .tests.test_correctness import aggregation_check, centralized_equivalence
from .utils import keyed_seed, read_json, sha256, write_csv, write_json


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def synthetic_signal(device):
    """Two balanced spatial signals; 20 private synthetic clients, separate validation."""
    rng = np.random.default_rng(802)
    result = {}
    for cid in range(20):
        y = np.tile(np.arange(2), 32)
        rng.shuffle(y)
        x = rng.normal(0, .03, (64, 28, 28)).astype(np.float32)
        x[y == 0, 4:12, 4:12] += .8
        x[y == 1, 16:24, 16:24] += .8
        tx, ty = torch.tensor(x, device=device), torch.tensor(y, device=device)
        result[cid] = ClientData(cid, tx[:48], ty[:48], tx[48:], ty[48:])
    return result


def permute_training_labels(data):
    result = {}
    for cid, c in data.items():
        rng = np.random.default_rng(keyed_seed(613, cid))
        order = torch.as_tensor(rng.permutation(c.n_train), device=c.train_y.device)
        result[cid] = ClientData(cid, c.train_x, c.train_y[order], c.val_x, c.val_y)
    return result


def controlled_partition(data, heterogeneous, output):
    """Isolated stress control: repartition TRAINING pool only, keep validation fixed.

    This centralized construction is only an experimental intervention, never the
    required user-partitioned training path or parameter aggregation interface.
    """
    x = torch.cat([c.train_x for c in data.values()])
    y = torch.cat([c.train_y for c in data.values()])
    rng = np.random.default_rng(774)
    order = rng.permutation(len(y))
    if heterogeneous:
        order = order[np.argsort(y.cpu().numpy()[order], kind="stable")]
    result, manifests, start = {}, {}, 0
    for cid, c in data.items():
        ids = order[start:start+c.n_train]
        index = torch.as_tensor(ids, device=y.device)
        result[cid] = ClientData(cid, x[index], y[index], c.val_x, c.val_y)
        manifests[str(cid)] = ids.tolist()
        start += c.n_train
    assert start == len(y) and np.array_equal(np.sort(order), np.arange(len(y)))
    write_json(output, dict(pool_order="Original client ID order then original training tensor order",
                           seed=774, heterogeneous=heterogeneous, client_training_indices=manifests))
    return result


def entropy_summary(data):
    ent = [label_statistics(c.train_y.cpu().numpy())[0]["entropy_bits"] for c in data.values()]
    return dict(mean_entropy_bits=float(np.mean(ent)), min_entropy_bits=float(np.min(ent)),
                max_entropy_bits=float(np.max(ent)))


def run_validations(args, data, manifest):
    root, results = Path(args.output_dir), []
    out = root/"validation"
    out.mkdir(parents=True, exist_ok=True)

    def record(code, hypothesis, setup, expected, observed, passed, interpretation, limitations, gating=False):
        r = dict(test=code, hypothesis=hypothesis, setup=setup, expected_behavior=expected,
                 observed=observed, passed=bool(passed), gating=gating,
                 interpretation=interpretation, limitations=limitations)
        results.append(r)
        write_json(out/"results.json", results)
        print("VALIDATION", code, "PASS" if passed else "FAIL", observed, flush=True)

    a = aggregation_check()
    record("A_exact_aggregation", "Unequal training counts produce the exact weighted mean",
           "Three deterministic float32 vectors; n=[1,3,6]", "Maximum error <= 1e-6", a,
           a["max_abs_error"] <= a["tolerance"], "Direct numerical verification of sample weighting",
           "Small deterministic tensors; end-to-end behavior tested separately", True)
    b = centralized_equivalence()
    record("B_centralized_equivalence", "One local full-batch step equals centralized full-batch SGD",
           "17 synthetic examples split 2/5/10; all participate; E=1; LR=.03; same MLP", "Max parameter error <= 3e-7", b,
           b["max_abs_error"] <= b["tolerance"], "Validates the gradient/parameter averaging identity",
           "Identity applies to one full-batch step, not multiple local mini-batches", True)
    sampling = []
    for fraction in [.02, .05, .10]:
        rng, second = np.random.default_rng(80), np.random.default_rng(80)
        counts = np.zeros(100, int)
        valid = True
        rounds = 10000
        k = int(100 * fraction)
        for _ in range(rounds):
            chosen = select_clients(list(range(100)), fraction, rng)
            valid &= len(chosen) == k and len(set(chosen)) == k
            valid &= chosen == select_clients(list(range(100)), fraction, second)
            counts[chosen] += 1
        expected = rounds * fraction
        sd = np.sqrt(rounds * fraction * (1-fraction))
        max_z = float(np.max(np.abs(counts-expected)) / sd)
        sampling.append(dict(fraction=fraction, rounds=rounds, expected_per_client=expected,
                             binomial_sd=sd, min=int(counts.min()), max=int(counts.max()),
                             max_abs_z=max_z, correct_unique_reproducible=bool(valid),
                             counts=counts.tolist()))
    record("C_sampling", "Each eligible client has probability C each round", "10,000 draws for each C; duplicate seeded stream",
           "Exact k, no duplicates, identical seeded sequences; all client counts within 5 binomial SD",
           sampling, all(s["correct_unique_reproducible"] and s["max_abs_z"] < 5 for s in sampling),
           "Participation matches expected marginal random variation", "Counts across clients are dependent within a round; diagnostic is not proof of uniformity", True)

    split_hash = sha256(root/"splits.json")
    synthetic = synthetic_signal(args.device)
    controlled = []
    for lr in [.0001, .1]:
        cfg = Config(25, .1, 1, lr, 16, 402, args.device, "synthetic-seed-802", "synthetic_spatial_binary")
        controlled.append(run_experiment(synthetic, cfg, out/f"signal_lr{lr}"))
    low, good = controlled
    history = read_csv(out/"signal_lr0.1/history.csv")
    start_loss, end_loss = float(history[0]["train_loss"]), float(history[-1]["train_loss"])
    evidence = dict(initial_train_loss=start_loss, final_train_loss=end_loss,
                    suitable_lr_accuracy=good["final_validation_accuracy"],
                    tiny_lr_accuracy=low["final_validation_accuracy"], suitable_lr=.1, tiny_lr=.0001)
    record("D_controlled_learning", "SGD learns a spatial signal; extremely small LR learns slower",
           "20 synthetic clients; 960 train / 320 validation; balanced two-class signal; same initial weights and selected clients",
           "LR .1: final validation >90%, loss <50% of initial; >20 percentage points above LR .0001",
           evidence, good["final_validation_accuracy"] > .9 and end_loss < .5*start_loss and good["final_validation_accuracy"] - low["final_validation_accuracy"] > .2,
           "Checks useful learning and a controlled optimization response", "Synthetic binary task is easier than 62-class handwriting", True)

    decision = read_json(root/"pilot_decision.json")
    base = Config(40, .1, 3, decision["lr"], 32, 2026, args.device, split_hash, "correct_labels_control")
    correct = run_experiment(data, base, out/"correct_labels")
    shuffled_data = permute_training_labels(data)
    shuffled = run_experiment(shuffled_data, replace(base, data_tag="within_client_shuffled_training_labels_seed613"), out/"shuffled_labels")
    gap = correct["final_validation_accuracy"] - shuffled["final_validation_accuracy"]
    record("E_label_randomization", "Destroying training-label information prevents meaningful validation learning",
           "Full development pool, 40 rounds, C=.1 E=3, matched init/sampling/batches/LR; permute labels within each client's training set; validation unchanged",
           "Correct-label final validation exceeds shuffled-label validation by >15 percentage points",
           dict(correct_validation=correct["final_validation_accuracy"], shuffled_validation=shuffled["final_validation_accuracy"], gap=gap),
           gap > .15, "Within-client permutation preserves label counts and client sizes while breaking image-label correspondence",
           "One matched seed; finite shuffled data can retain accidental correlations; does not prove privacy", True)
    del shuffled_data

    partition_results = []
    for heterogeneous, tag in [(False, "randomized_homogeneous"), (True, "label_sorted_heterogeneous")]:
        partitioned = controlled_partition(data, heterogeneous, out/f"{tag}_partition.json")
        summary = run_experiment(partitioned, replace(base, data_tag=tag), out/tag)
        partition_results.append(dict(partition=tag, **entropy_summary(partitioned),
                                      final_validation_accuracy=summary["final_validation_accuracy"],
                                      best_validation_accuracy=summary["best_validation_accuracy"],
                                      train_samples=sum(c.n_train for c in partitioned.values()),
                                      cumulative_examples_seen=summary["cumulative_examples_seen"]))
        del partitioned
    original_ent = entropy_summary(data)
    observed = dict(original=original_ent, controlled=partition_results,
                    homogeneous_minus_heterogeneous=partition_results[0]["final_validation_accuracy"]-partition_results[1]["final_validation_accuracy"])
    record("F_heterogeneity", "Stronger label segregation creates client drift under several local SGD steps",
           "Reassign only the 25,418 TRAINING examples into original size vector; shuffled vs label-sorted assignment; unchanged validation; 40 rounds C=.1 E=3",
           "Sorted partition has lower entropy; examine validation gap rather than impose a universal ordering",
           observed, partition_results[1]["mean_entropy_bits"] < partition_results[0]["mean_entropy_bits"],
           "This controlled intervention isolates partition structure at matched sizes, counts, rounds and sampling; sample exposure is equal",
           "Extreme artificial partitions; original user partitions remain required main results; label entropy does not measure writer-style heterogeneity")

    try:
        stress = run_experiment(synthetic, Config(5, .1, 3, 1e6, 16, 402, args.device,
                                "synthetic-seed-802", "excessive_lr_stress"), out/"excessive_lr")
        bad_history = read_csv(out/"excessive_lr/history.csv")
        bad_loss = float(bad_history[-1]["train_loss"])
        evidence = dict(nonfinite_rejected=False, final_train_loss=bad_loss,
                        suitable_lr_final_train_loss=end_loss)
        passed = bad_loss > end_loss * 10
    except FloatingPointError as error:
        evidence, passed = dict(nonfinite_rejected=True, error=str(error)), True
        write_json(out/"excessive_lr/rejection.json", evidence)
    record("edge_excessive_lr", "Excessive LR destabilizes learning and nonfinite values are rejected",
           "Same synthetic signal; LR=1e6; 5 rounds E=3", "Clear deterioration or explicit FloatingPointError", evidence,
           passed, "No failed update is substituted or silently included in averaging", "Synthetic stress case, not an observed source-data failure", True)

    rows = read_json(root/"comparison.json")
    repeat_stats = []
    for c, e in sorted({(r["fraction"], r["epochs"]) for r in rows}):
        group = [r for r in rows if r["fraction"] == c and r["epochs"] == e]
        if len(group) < 3:
            continue
        vals = [r["best_validation_accuracy"] for r in group]
        finals = [r["final_validation_accuracy"] for r in group]
        repeat_stats.append(dict(fraction=c, epochs=e, seeds=[r["seed"] for r in group],
                                 best_mean=float(np.mean(vals)), best_sd=float(np.std(vals, ddof=1)),
                                 final_mean=float(np.mean(finals)), final_sd=float(np.std(finals, ddof=1)),
                                 best_min=min(vals), best_max=max(vals)))
    primary = [r for r in rows if r["seed"] == 2026]
    same_init = len({r["initial_digest"] for r in primary}) == 1
    record("G_reproducibility", "Leading settings should retain useful validation performance across seeds",
           "Top two primary-seed settings repeated with seeds 2027,2028; split seed stays 490",
           "Identical initialization within seed; quantify cross-seed variation", dict(identical_primary_initialization=same_init, repeats=repeat_stats),
           same_init and len(repeat_stats) == 2, "Reports sample SD across three seeds instead of treating one trajectory as definitive",
           "Three seeds are insufficient for narrow population uncertainty; seed changes init, selection and minibatch order", True)

    # Observational per-client benefit and size/entropy relationships for baseline.
    baseline = root/"experiments/c10_e1_s2026"
    initial = {int(r["client_id"]): r for r in read_csv(baseline/"client_initial.csv")}
    final = {int(r["client_id"]): r for r in read_csv(baseline/"client_final.csv")}
    frequencies = np.zeros(100, int)
    for r in read_csv(baseline/"selections.csv"):
        frequencies[int(r["client_id"])] += 1
    client_stats = {int(r["client_id"]): r for r in read_csv(root/"analysis/client_statistics.csv")}
    benefit_rows = []
    for cid in sorted(data):
        stats = client_stats[cid]
        benefit_rows.append(dict(client_id=cid, participation=int(frequencies[cid]), train_n=int(stats["train_n"]),
                                 entropy_bits=float(stats["entropy_bits"]), observed_classes=int(stats["observed_classes"]),
                                 initial_accuracy=float(initial[cid]["accuracy"]), final_accuracy=float(final[cid]["accuracy"]),
                                 benefit=float(final[cid]["accuracy"])-float(initial[cid]["accuracy"])))
    write_csv(out/"client_benefits.csv", benefit_rows)
    correlations = {}
    benefits = np.array([r["benefit"] for r in benefit_rows])
    for field in ["participation", "train_n", "entropy_bits"]:
        correlations[field] = float(np.corrcoef([r[field] for r in benefit_rows], benefits)[0, 1])
    ordered = sorted(benefit_rows, key=lambda r: (r["benefit"], r["client_id"]))
    groups = {}
    for field in ["train_n", "entropy_bits"]:
        sorted_rows = sorted(benefit_rows, key=lambda r: (r[field], r["client_id"]))
        groups[field] = dict(lowest_quartile_mean_benefit=float(np.mean([r["benefit"] for r in sorted_rows[:25]])),
                             highest_quartile_mean_benefit=float(np.mean([r["benefit"] for r in sorted_rows[-25:]])))
    record("F_client_imbalance_observational", "Client size and heterogeneity may accompany uneven benefits",
           "100 original clients; baseline final-round validation minus round-zero validation", "Describe correlations and extremes, without causal claims",
           dict(correlations=correlations, quartile_groups=groups, least_benefit=ordered[:3], most_benefit=ordered[-3:],
                participation_min=int(frequencies.min()), participation_max=int(frequencies.max()), expected_participation=8.),
           True, "Local validation sets differ in size and class mix; sampling frequency is a random exposure measure",
           "Observational associations are not causal; small client validation sets have substantial binomial noise")
    write_json(out/"completion.json", dict(n_investigations=len(results), all_gating_passed=all(r["passed"] for r in results if r["gating"])))
