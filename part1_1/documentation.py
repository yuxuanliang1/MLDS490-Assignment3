"""Generate report notes from measured artifacts; never contains guessed metrics."""
from pathlib import Path
import csv

import numpy as np

from .utils import read_json, write_json


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def percent(v):
    return f"{100*v:.2f}%"


def generate(root):
    root = Path(root)
    data = read_json(root/"analysis/dataset_summary.json")
    env = read_json(root/"environment.json")
    pilot = read_json(root/"pilot_decision.json")
    comparison = read_json(root/"comparison.json")
    selected = read_json(root/"selection.json")
    final = read_json(root/"final_test.json")
    validations = read_json(root/"validation/results.json")
    unit = read_json(root/"validation/correctness_tests.json")
    baseline = next(r for r in comparison if r["name"] == "c10_e1_s2026")
    baseline_history = csv_rows(root/"experiments/c10_e1_s2026/history.csv")
    init, last = baseline_history[0], baseline_history[-1]
    sizes = data["size_statistics"]
    notes = ["# Assignment 3, Part 1.1: sequential FedAvg report notes", "",
             "## Scope and evidence", "",
             "All final reported training, scientific experiments, correctness reruns, plots and held-out evaluation were executed on DeepDish5. Part 1.2 (Ray) and Part 2 (differential privacy) are intentionally unimplemented. Earlier local exploratory pilots remain in `part1_1_outputs/` and are excluded from every comparison and selection reported here.", "",
             "The assignment PDF (general requirements on pages 1-2; Part 1.1 on pages 2-3) requires edge-case evidence, validation experiments, a non-standard plot, reproducible instructions, and a three-month extension plan. It allocates 75% to edge cases/tests. The Week 7 lecture (FedAvg slides 20-27, heterogeneity slides 29-36) and Week 7 lab (slides 2-4) motivate local SGD with training-count-weighted model averaging. Submission still requires the user's GitHub username, public repository URL, and a final combined PDF report; these notes are the Part 1.1 contribution, not that later combined submission.", "",
             "## Dataset and preprocessing", "",
             "`Assignment3-data/train_data.npy` is a trusted NumPy object array of shape `(100,)`. Each indexed dictionary has exactly `images` and `labels`; the original client ID is its array index. Values are lists containing float32 28 x 28 images and int32 labels. IDs run from 0 through 99. The data README identifies a LEAF-derived subset and a separately aggregated test file. No replacement/downloaded dataset was used.", "",
             f"The development pool has **{data['pool_examples']:,} examples**, **{data['clients']} clients**, and **{data['classes']} observed class IDs (0-61)**. Client sample sizes: minimum {sizes['min']}, first quartile {sizes['quartiles'][0]:g}, median {sizes['median']:g}, third quartile {sizes['quartiles'][1]:g}, maximum {sizes['max']}, mean {sizes['mean']:.2f}. All {data['eligible_clients']} clients have training data and participate in the eligibility pool.", "",
             "The source files provide no verified numeric-label-to-character lookup, so plots retain numeric IDs. The assignment describes 10 digits, 26 uppercase and 26 lowercase letters; no ordering is inferred from that description. This avoids fabricating class names.", "",
             "Pixels are already finite float32 values in [0,1]. Visual inspection of development images shows dark, small glyphs on white backgrounds. The fixed transformation is **x' = 1 - x** (zero background, positive ink). There is no resizing, augmentation, rotation, transpose, or learned normalization. Labels are cast to int64 for CrossEntropyLoss. The same transformation is applied during final test evaluation. The original images and labels are never modified on disk.", "",
             f"Each client is independently permuted with a SHA-256-derived seed from `(490, client_id)`. Training receives floor(0.8 n) examples and validation the remainder; all {data['training_examples']:,} training and {data['validation_examples']:,} validation indices are saved in `splits.json` and reused unchanged across every main experiment. Coverage and disjointness are asserted. Splits are random, not stratified; classes missing from a small client or subset remain missing. Source SHA-256: `{data['source_sha256']}`.", "",
             "Tiny-client policy: n=0 is ineligible; n=1 is training-only with empty validation; n>=2 has disjoint nonempty subsets (clamped floor policy). Empty local validation contributes zero metric sums; no fabricated examples or duplicated indices are introduced. If the aggregate validation partition is empty, model selection raises an error. These tiny/empty cases are synthetic stress tests: none occurs in the supplied 100 clients.", "",
             "## Algorithm and logical data boundaries", "",
             "The model is exactly Flatten -> Linear(784,128) -> ReLU -> Linear(128,62), with 108,478 trainable float32 parameters and raw logits. Local optimization uses mean CrossEntropyLoss, shuffled mini-batches of 32, plain SGD, no momentum, no weight decay, and a freshly created model/optimizer for every client update.", "",
             r"For selected clients $S_t$, the update is $w_{t+1}=\sum_{k\in S_t}\frac{n_k^{\mathrm{train}}}{\sum_{j\in S_t}n_j^{\mathrm{train}}}w_{t+1}^{(k)}$.", "",
             "Each client starts from the identical cloned current global state and performs E complete local epochs, including its final short batch. The server receives only model parameters, client ID, training count and step metadata. Aggregation sees no raw images/labels. Clients evaluate the global model locally; the server sums cross-entropy totals, correct counts and sample counts. Thus global loss = total CE / total examples and accuracy = total correct / total examples. It never averages unweighted client accuracies. Every round evaluates the UPDATED model on ALL original clients' train and validation data, including nonparticipants; round zero is saved too.", "",
             "Clients are uniformly sampled without replacement. The main sampling sizes are exactly 2, 5 and 10 for C=.02,.05,.10. Distinct C values have independent keyed sampling streams; E comparisons at fixed C share selections. Local shuffle streams depend on seed, round and client. Within a seed all six settings have the same initial-weight digest. Source data can reside on one physical server, but the parameter-only training/aggregation boundary remains explicit. Only the isolated partition-intervention experiment constructs new development training partitions.", "",
             "Nonfinite local losses, nonfinite parameter updates, invalid sample counts, duplicate client IDs, and incompatible tensor shapes/types are rejected explicitly. The entire failed round is not committed. There is no silent failed-client substitution. Atomic last-round checkpoints contain sampling RNG state, history and selections; interruption/resume is validated against an uninterrupted run.", "",
             "## DeepDish5 environment and runtime", "",
             f"Host `{env['hostname']}`; Python `{env['python'].split()[0]}` at `{env['executable']}`; PyTorch `{env['torch']}`, CUDA build `{env['cuda_build']}`, NumPy `{env['numpy']}`, Matplotlib `{env['matplotlib']}`. GPU: **{env['gpu']}**, compute capability {env['gpu_compute_capability']}, {env['gpu_memory_bytes']/2**30:.2f} GiB. Hardware-supported compiled architectures: `{env['cuda_architectures']}`. A real CUDA float32 forward/backward probe succeeded before training.", "",
             f"The server exposes {env['logical_cpus']} logical CPUs and {env['ram_bytes']/2**30:.1f} GiB RAM. Two PyTorch CPU threads were used. Physical GPU 0 was idle at launch; the busy RTX PRO GPU and other users' processes were left alone. `CUDA_VISIBLE_DEVICES` uses physical GPU 0's UUID, with PCI ordering, to avoid CUDA's default fastest-device enumeration. All client updates run sequentially. Deterministic algorithms are enabled, CUBLAS workspace is fixed, and TF32 is disabled.", "",
             f"Baseline loop runtime was {baseline['elapsed_seconds']:.2f} s; peak allocated PyTorch CUDA memory {baseline['peak_cuda_memory_bytes']/2**20:.1f} MiB. The ten comparison/repeat runs total {sum(r['elapsed_seconds'] for r in comparison):.2f} s of measured loop time. Pilot runtimes total {sum(r['elapsed_seconds'] for r in pilot['pilots']):.2f} s; the pilot estimated the six-run grid at {pilot['estimated_grid_seconds']:.2f} s. These timers include round metrics/checkpoints after initial evaluation, but exclude SSH/transfers, Python startup, and initial round-zero evaluation. Dedicated JSON environment and nvidia-smi logs preserve current hardware and utilization evidence. Runtime does not imply isolated-GPU throughput or a network communication benchmark.", "",
             "## Pilot and fixed comparison protocol", "",
             "After the local exploration was stopped, all six pilot settings were rerun on DeepDish5 with the fixed inverted-pixel preprocessing. They use 12 rounds, C=.1, E in {1,3}, LR in {.01,.1,.3}. The decision rule, applied before the main grid, chooses the LR with largest mean FINAL round-12 validation accuracy over the two E values. Smaller LR breaks ties. Test data is unopened during this process.", "",
             "| LR | Mean round-12 validation accuracy over E=1,3 |", "|---:|---:|"]
    for lr, v in pilot["mean_validation_by_lr"].items():
        notes.append(f"| {lr} | {percent(v)} |")
    notes += ["", f"Selected shared LR = **{pilot['lr']}**. The suggested 80-round budget and batch size 32 were retained for all six main settings. All data are used; no main experiment uses subsampling. Fixed rounds make the comparison interpretable but do not establish convergence to an optimum.", "",
              "## Baseline and C/E comparison", "",
              f"Baseline C=.10, E=1: round-zero train/validation accuracy {percent(float(init['train_accuracy']))} / {percent(float(init['validation_accuracy']))}; round-80 train/validation accuracy **{percent(baseline['final_train_accuracy'])} / {percent(baseline['final_validation_accuracy'])}**. Training CE decreases from {float(init['train_loss']):.6f} to {float(last['train_loss']):.6f}, validation CE from {float(init['validation_loss']):.6f} to {float(last['validation_loss']):.6f}. Best validation accuracy is {percent(baseline['best_validation_accuracy'])} at round {baseline['best_round']}.", "",
              "| C | E | Clients/round | Final train | Final validation | Best validation (round) | Client-epochs | Example exposures | Loop seconds |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparison:
        if r["seed"] == 2026:
            notes.append(f"| {r['fraction']:.2f} | {r['epochs']} | {r['clients_per_round']} | {percent(r['final_train_accuracy'])} | {percent(r['final_validation_accuracy'])} | {percent(r['best_validation_accuracy'])} ({r['best_round']}) | {r['cumulative_client_epochs']} | {r['cumulative_examples_seen']:,} | {r['elapsed_seconds']:.2f} |")
    diagnostics=[]
    for r in comparison:
        if r["seed"] != 2026:
            continue
        trajectory=csv_rows(root/"experiments"/r["name"]/"history.csv")
        accuracies=np.array([float(h["validation_accuracy"]) for h in trajectory])
        diagnostics.append(dict(fraction=r["fraction"],epochs=r["epochs"],
                            last10_validation_sd_pp=float(100*np.std(accuracies[-10:],ddof=1)),
                            maximum_one_round_drop_pp=float(max(0,-100*np.diff(accuracies).min()))))
    from .utils import write_csv
    write_csv(root/"convergence_diagnostics.csv",diagnostics)
    notes += ["", "| C | E | Last 10 rounds validation sample SD (pp) | Largest one-round validation drop (pp) |",
              "|---:|---:|---:|---:|"]
    for r in diagnostics:
        notes.append(f"| {r['fraction']:.2f} | {r['epochs']} | {r['last10_validation_sd_pp']:.3f} | {r['maximum_one_round_drop_pp']:.3f} |")
    notes += ["", "These trajectory SDs include both learning trends and sampling fluctuations; they are stability diagnostics, not independent-run uncertainty estimates. The largest drop includes all consecutive pairs from round zero onward.", "",
              "E=3 generally supplies more local computation and data exposure per communication round; higher C also reduces aggregation sampling noise while exposing more training data. Compare measured curves rather than assuming either universally improves accuracy. The cost plot uses cumulative client-epochs, while CSVs also record actual sample exposures and SGD steps. Equal rounds span 160 to 2,400 client-epochs, a 15-fold difference. One model contains 433,912 bytes of float32 parameters; an idealized upload-plus-download proxy is 2 x clients-per-round x rounds x that size, excluding transport overhead and metadata. This simulation does not measure actual distributed communication latency.", "",
              "The two settings with highest best validation accuracy in seed 2026 were repeated in seeds 2027 and 2028. The split never changes. Across-seed variability therefore combines initialization, participation and mini-batch ordering. These are sample SDs across only three seeds, not confidence intervals.", "",
              "| C | E | Seeds | Mean best validation +/- sample SD | Mean final validation +/- sample SD |", "|---:|---:|---:|---:|---:|"]
    for r in selected["repeated_summary"]:
        notes.append(f"| {r['fraction']:.2f} | {r['epochs']} | {r['n_seeds']} | {percent(r['mean_best_validation'])} +/- {100*r['sd_best_validation']:.2f} pp | {percent(r['mean_final_validation'])} +/- {100*r['sd_final_validation']:.2f} pp |")
    notes += ["", "## Scientific validation and edge-case evidence", "",
              f"The existing correctness suite was reused, extended for source-list emptiness and extreme imbalance, and rerun on DeepDish5: **{unit['tests_run']} tests, {unit['failures']} failures, {unit['errors']} errors**. The tests cover exact aggregation, one-step centralized equivalence, tiny/empty splits, sampling, bad updates, broadcast isolation/fresh optimizer repeatability, empty client handling, sample-weighted metrics, invalid pixels/labels/C, restart identity, empty input lists, a 1:99,999 aggregation-count ratio, and explicit nonfinite local-loss rejection. The empty-list loader correction does not change the nonempty supplied data path or any completed main result. Scientific evidence below extends beyond these unit checks.", ""]
    for r in validations:
        notes += [f"### {r['test']}", "", f"**Hypothesis:** {r['hypothesis']}", "", f"**Setup:** {r['setup']}", "",
                  f"**Expected behavior / criterion:** {r['expected_behavior']}", "", f"**Observed outcome:** {'PASS' if r['passed'] else 'FAIL'}.", "", "```json"]
        import json
        # Do not flood notes with 300 sampling counts: they remain in the JSON evidence.
        observed = r["observed"]
        if r["test"] == "C_sampling":
            observed = [{k:v for k,v in row.items() if k != "counts"} for row in observed]
        notes += [json.dumps(observed, indent=2), "```", "", f"**Interpretation:** {r['interpretation']}", "", f"**Limitations:** {r['limitations']}", ""]
    notes += ["## Representative clients and data heterogeneity", "",
              "Five clients were selected systematically from training-only summaries. Ties use smaller client IDs and selected IDs are unique. The typical client minimizes standardized squared distance to medians of training size, observed-class count and entropy. Entropy is Shannon entropy in bits; all counts in this table are post-split training counts.", "",
              "| Client | Criterion | Training examples | Observed classes | Dominant class fraction | Entropy (bits) |",
              "|---:|---|---:|---:|---:|---:|"]
    for r in data["representative_clients"]:
        notes.append(f"| {r['client_id']} | {r['reason']} | {r['train_n']} | {r['observed_classes']} | {r['dominant_proportion']:.4f} | {r['entropy_bits']:.4f} |")
    counts = read_json(root/"analysis/class_counts.json")["development_pool"]
    dominant, rarest = int(np.argmax(counts)), int(np.argmin(counts))
    stats = csv_rows(root/"analysis/client_statistics.csv")
    notes += ["", f"Development counts range from {counts[rarest]} for ID {rarest} to {counts[dominant]} for ID {dominant}; the largest/smallest frequency ratio is {max(counts)/min(counts):.2f}. Original clients' training class counts range from {min(int(r['observed_classes']) for r in stats)} to {max(int(r['observed_classes']) for r in stats)}; training entropy ranges from {min(float(r['entropy_bits']) for r in stats):.4f} to {max(float(r['entropy_bits']) for r in stats):.4f} bits. All clients lack at least some classes. Size differences and writer identity provide heterogeneity even where class entropies are similar. Overall class imbalance must not be confused with between-client heterogeneity.", "",
              "## Figures and non-standard diagnostic", ""]
    for r in read_json(root/"figures/manifest.json"):
        notes.append(f"- `{r['name']}.png` / `.pdf`: {r['description']}.")
    obs = next(r["observed"] for r in validations if r["test"] == "F_client_imbalance_observational")
    notes += ["", "The required non-standard plot is **Client Participation, Heterogeneity, and Learning Benefit**. X is baseline participation count, Y is final-minus-initial client validation accuracy in percentage points, color is training-label entropy, and point area represents training size. It includes a zero-benefit line, the expected count 8 line, colorbar, size legend, and labels on benefit extremes.", "",
              f"Actual participation spans {obs['participation_min']} to {obs['participation_max']} rounds. Pearson correlations of benefit with participation, size and entropy are {obs['correlations']['participation']:.3f}, {obs['correlations']['train_n']:.3f}, {obs['correlations']['entropy_bits']:.3f}. Low-benefit IDs are {[r['client_id'] for r in obs['least_benefit']]}; high-benefit IDs are {[r['client_id'] for r in obs['most_benefit']]}. These points describe uneven client outcomes, not proven effects of participation. Local validation sample size, starting accuracy, class mix and writer difficulty can explain extremes; the saved per-client CSV supports follow-up. The quartile summaries above quantify small/large and low/high entropy groups without causal claims.", "",
              "## Frozen selection and single held-out evaluation", "",
              selected["rule"] + ".", "",
              f"Chosen C={selected['configuration']['fraction']:.2f}, E={selected['configuration']['epochs']}, LR={selected['configuration']['lr']}, batch={selected['configuration']['batch_size']}, seed={selected['configuration']['seed']}, checkpoint round={selected['selected_round']}. Best selected validation accuracy: **{percent(selected['configuration']['best_validation_accuracy'])}**. The complete selection and checkpoint SHA-256 were saved before opening test data.", "",
              f"The held-out file is a NumPy object array of shape {final['test_container_shape']} with one aggregated dictionary, images shape {final['image_shape']}, source labels dtype {final['label_dtype']}. It contains **{final['n']:,} test examples**. The frozen checkpoint obtains **{percent(final['accuracy'])} test accuracy ({final['correct']:,}/{final['n']:,} correct)** and test CE {final['loss']:.6f}. This is one aggregate test result; per-writer test attribution is unavailable. Test results were not used to revise the configuration, number of rounds, normalization, checkpoint or seed. The result is cached against the selection hash to prevent accidental rescreening.", "",
              "## Limitations", "",
              "- A fixed 80-round budget and this small MLP do not establish best attainable EMNIST performance. Rare classes and ambiguous uppercase/lowercase characters can remain difficult; numeric IDs alone limit semantic error analysis.",
              "- Three seeds only for the top two configurations provide limited uncertainty evidence. Validation-based selection over settings, checkpoints and seeds has selection optimism. The held-out result is the independent final check; do not tune further against it.",
              "- The test set is aggregated, so sample-wise accuracy cannot establish writer-level fairness, out-of-client generalization, or client-specific test uncertainty. Label entropy measures only one dimension of non-IID data.",
              "- Artificial homogeneous/heterogeneous partitions are controlled interventions, not replacements for real client boundaries. The label-randomization experiment is a learning sanity check, not a statistical privacy guarantee.",
              "- This is sequential synchronous simulation on one GPU. Measured loop time includes evaluation/I/O and is not a distributed latency or energy benchmark. Other server activity can affect runtime. Exact bitwise reproduction across hardware or PyTorch releases is not guaranteed.", "",
              "## Proposed three-month extended validation plan", "",
              "1. **Month 1: reliability and uncertainty.** Freeze this final test evaluation. Add 10-20 prespecified seeds for all six settings, bootstrap validation at the client level, collect rare-class recall/confusion and calibration metrics using a verified character mapping. Repeat preprocessing and LR ablations on development data with fixed search budgets. Validate data lineage, resumption after process termination and CPU/CUDA numerical agreement.",
              "2. **Month 2: heterogeneity and robustness.** Run matched-size Dirichlet label partitions over a prespecified concentration grid, writer-style/quantity skew experiments, controlled missing classes, singleton and highly unequal clients, client drift norms, and sample-versus-client fairness summaries. Evaluate dropout, stragglers and corruption with explicit failure policies. Compare fixed-round, fixed-client-epoch and fixed-example-exposure budgets; keep original writer partitions as the principal setting.",
              "3. **Month 3: confirmatory generalization and systems.** Reserve a newly held-out writer-level development split for confirmatory protocol selection, pre-register comparisons, then obtain a fresh untouched final test source before any new final scoring. Measure repeated hardware runtime/memory/energy and communication proxies with confidence intervals. Analyze accuracy/fairness tradeoffs, document failure cases, audit all plots against source metrics, and write a reproducible artifact package. Ray and privacy comparisons belong to their later separately scoped assignment parts and are not claimed here.", ""]
    (root/"REPORT_NOTES.md").write_text("\n".join(notes), encoding="utf-8")
    write_json(root/"completion_summary.json", dict(dataset=data, baseline=baseline,
               selection=selected, final_test=final, correctness=unit,
               validation_investigations=len(validations), all_validation_passed=all(r["passed"] for r in validations),
               comparison_runs=len(comparison), figure_pairs=len(read_json(root/"figures/manifest.json"))))
