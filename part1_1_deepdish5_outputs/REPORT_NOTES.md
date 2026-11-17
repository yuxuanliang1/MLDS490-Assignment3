# Assignment 3, Part 1.1: sequential FedAvg report notes

## Scope and evidence

All final reported training, scientific experiments, correctness reruns, plots and held-out evaluation were executed on DeepDish5. The completed [Ray](../part1_2/README.md) and [Laplace](../part2/README.md) components reuse this baseline; the combined [report](../report/README.md) integrates all three parts. Earlier local exploratory pilots remain in `part1_1_outputs/` and are excluded from every comparison and selection reported here.

The assignment PDF (general requirements on pages 1-2; Part 1.1 on pages 2-3) requires edge-case evidence, validation experiments, a non-standard plot, reproducible instructions, and a three-month extension plan. It allocates 75% to edge cases/tests. The Week 7 lecture (FedAvg slides 20-27, heterogeneity slides 29-36) and Week 7 lab (slides 2-4) motivate local SGD with training-count-weighted model averaging. The combined report includes the GitHub identifiers and reproduction instructions; these notes retain the detailed Part 1.1 evidence.

## Dataset and preprocessing

`Assignment3-data/train_data.npy` is a trusted NumPy object array of shape `(100,)`. Each indexed dictionary has exactly `images` and `labels`; the original client ID is its array index. Values are lists containing float32 28 x 28 images and int32 labels. IDs run from 0 through 99. The data README identifies a LEAF-derived subset and a separately aggregated test file. No replacement/downloaded dataset was used.

The development pool has **31,825 examples**, **100 clients**, and **62 observed class IDs (0-61)**. Client sample sizes: minimum 118, first quartile 285, median 331, third quartile 351.25, maximum 393, mean 318.25. All 100 clients have training data and participate in the eligibility pool.

The source files provide no verified numeric-label-to-character lookup, so plots retain numeric IDs. The assignment describes 10 digits, 26 uppercase and 26 lowercase letters; no ordering is inferred from that description. This avoids fabricating class names.

Pixels are already finite float32 values in [0,1]. Visual inspection of development images shows dark, small glyphs on white backgrounds. The fixed transformation is **x' = 1 - x** (zero background, positive ink). There is no resizing, augmentation, rotation, transpose, or learned normalization. Labels are cast to int64 for CrossEntropyLoss. The same transformation is applied during final test evaluation. The original images and labels are never modified on disk.

Each client is independently permuted with a SHA-256-derived seed from `(490, client_id)`. Training receives floor(0.8 n) examples and validation the remainder; all 25,418 training and 6,407 validation indices are saved in `splits.json` and reused unchanged across every main experiment. Coverage and disjointness are asserted. Splits are random, not stratified; classes missing from a small client or subset remain missing. Source SHA-256: `9afa6ee4eff168daf080e73381fb6392ade8f970daf4392183f4c1f6c7db4aec`.

Tiny-client policy: n=0 is ineligible; n=1 is training-only with empty validation; n>=2 has disjoint nonempty subsets (clamped floor policy). Empty local validation contributes zero metric sums; no fabricated examples or duplicated indices are introduced. If the aggregate validation partition is empty, model selection raises an error. These tiny/empty cases are synthetic stress tests: none occurs in the supplied 100 clients.

## Algorithm and logical data boundaries

The model is exactly Flatten -> Linear(784,128) -> ReLU -> Linear(128,62), with 108,478 trainable float32 parameters and raw logits. Local optimization uses mean CrossEntropyLoss, shuffled mini-batches of 32, plain SGD, no momentum, no weight decay, and a freshly created model/optimizer for every client update.

For selected clients $S_t$, the update is $w_{t+1}=\sum_{k\in S_t}\frac{n_k^{\mathrm{train}}}{\sum_{j\in S_t}n_j^{\mathrm{train}}}w_{t+1}^{(k)}$.

Each client starts from the identical cloned current global state and performs E complete local epochs, including its final short batch. The server receives only model parameters, client ID, training count and step metadata. Aggregation sees no raw images/labels. Clients evaluate the global model locally; the server sums cross-entropy totals, correct counts and sample counts. Thus global loss = total CE / total examples and accuracy = total correct / total examples. It never averages unweighted client accuracies. Every round evaluates the UPDATED model on ALL original clients' train and validation data, including nonparticipants; round zero is saved too.

Clients are uniformly sampled without replacement. The main sampling sizes are exactly 2, 5 and 10 for C=.02,.05,.10. Distinct C values have independent keyed sampling streams; E comparisons at fixed C share selections. Local shuffle streams depend on seed, round and client. Within a seed all six settings have the same initial-weight digest. Source data can reside on one physical server, but the parameter-only training/aggregation boundary remains explicit. Only the isolated partition-intervention experiment constructs new development training partitions.

Nonfinite local losses, nonfinite parameter updates, invalid sample counts, duplicate client IDs, and incompatible tensor shapes/types are rejected explicitly. The entire failed round is not committed. There is no silent failed-client substitution. Atomic last-round checkpoints contain sampling RNG state, history and selections; interruption/resume is validated against an uninterrupted run.

## DeepDish5 environment and runtime

Host `mlds-deepdish5`; Python `3.12.6`; PyTorch `2.14.0+cu130`, CUDA build `13.0`, NumPy `2.5.3`, Matplotlib `3.11.2`. GPU: **NVIDIA GeForce RTX 5060 Ti**, compute capability [12, 0], 15.57 GiB. Hardware-supported compiled architectures: `['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`. A real CUDA float32 forward/backward probe succeeded before training.

The server exposes 48 logical CPUs and 93.8 GiB RAM. Two PyTorch CPU threads were used. Physical GPU 0 was idle at launch; the busy RTX PRO GPU and other users' processes were left alone. `CUDA_VISIBLE_DEVICES` uses physical GPU 0's UUID, with PCI ordering, to avoid CUDA's default fastest-device enumeration. All client updates run sequentially. Deterministic algorithms are enabled, CUBLAS workspace is fixed, and TF32 is disabled.

Baseline loop runtime was 9.21 s; peak allocated PyTorch CUDA memory 166.8 MiB. The ten comparison/repeat runs total 104.49 s of measured loop time. Pilot runtimes total 16.64 s; the pilot estimated the six-run grid at 110.95 s. These timers include round metrics/checkpoints after initial evaluation, but exclude SSH/transfers, Python startup, and initial round-zero evaluation. Dedicated JSON environment and nvidia-smi logs preserve current hardware and utilization evidence. Runtime does not imply isolated-GPU throughput or a network communication benchmark.

## Pilot and fixed comparison protocol

After the local exploration was stopped, all six pilot settings were rerun on DeepDish5 with the fixed inverted-pixel preprocessing. They use 12 rounds, C=.1, E in {1,3}, LR in {.01,.1,.3}. The decision rule, applied before the main grid, chooses the LR with largest mean FINAL round-12 validation accuracy over the two E values. Smaller LR breaks ties. Test data is unopened during this process.

| LR | Mean round-12 validation accuracy over E=1,3 |
|---:|---:|
| 0.01 | 6.86% |
| 0.1 | 28.33% |
| 0.3 | 46.69% |

Selected shared LR = **0.3**. The suggested 80-round budget and batch size 32 were retained for all six main settings. All data are used; no main experiment uses subsampling. Fixed rounds make the comparison interpretable but do not establish convergence to an optimum.

## Baseline and C/E comparison

Baseline C=.10, E=1: round-zero train/validation accuracy 3.19% / 3.03%; round-80 train/validation accuracy **67.02% / 65.29%**. Training CE decreases from 4.126150 to 1.246401, validation CE from 4.126700 to 1.328689. Best validation accuracy is 65.52% at round 78.

| C | E | Clients/round | Final train | Final validation | Best validation (round) | Client-epochs | Example exposures | Loop seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 1 | 2 | 60.83% | 59.03% | 61.28% (78) | 160 | 40,949 | 6.51 |
| 0.02 | 3 | 2 | 69.99% | 67.64% | 69.39% (77) | 480 | 122,847 | 7.81 |
| 0.05 | 1 | 5 | 66.40% | 64.66% | 64.66% (80) | 400 | 102,161 | 7.42 |
| 0.05 | 3 | 5 | 74.85% | 72.59% | 72.98% (78) | 1200 | 306,483 | 9.79 |
| 0.10 | 1 | 10 | 67.02% | 65.29% | 65.52% (78) | 800 | 203,782 | 9.21 |
| 0.10 | 3 | 10 | 75.32% | 73.00% | 73.00% (80) | 2400 | 611,346 | 13.80 |

| C | E | Last 10 rounds validation sample SD (pp) | Largest one-round validation drop (pp) |
|---:|---:|---:|---:|
| 0.02 | 1 | 6.047 | 20.259 |
| 0.02 | 3 | 1.418 | 9.630 |
| 0.05 | 1 | 1.154 | 2.747 |
| 0.05 | 3 | 1.301 | 3.231 |
| 0.10 | 1 | 0.717 | 7.133 |
| 0.10 | 3 | 0.598 | 1.389 |

These trajectory SDs include both learning trends and sampling fluctuations; they are stability diagnostics, not independent-run uncertainty estimates. The largest drop includes all consecutive pairs from round zero onward.

E=3 generally supplies more local computation and data exposure per communication round; higher C also reduces aggregation sampling noise while exposing more training data. Compare measured curves rather than assuming either universally improves accuracy. The cost plot uses cumulative client-epochs, while CSVs also record actual sample exposures and SGD steps. Equal rounds span 160 to 2,400 client-epochs, a 15-fold difference. One model contains 433,912 bytes of float32 parameters; an idealized upload-plus-download proxy is 2 x clients-per-round x rounds x that size, excluding transport overhead and metadata. This simulation does not measure actual distributed communication latency.

The two settings with highest best validation accuracy in seed 2026 were repeated in seeds 2027 and 2028. The split never changes. Across-seed variability therefore combines initialization, participation and mini-batch ordering. These are sample SDs across only three seeds, not confidence intervals.

| C | E | Seeds | Mean best validation +/- sample SD | Mean final validation +/- sample SD |
|---:|---:|---:|---:|---:|
| 0.05 | 3 | 3 | 71.94% +/- 0.99 pp | 71.52% +/- 1.14 pp |
| 0.10 | 3 | 3 | 72.77% +/- 0.23 pp | 72.40% +/- 0.57 pp |

## Scientific validation and edge-case evidence

The existing correctness suite was reused, extended for source-list emptiness and extreme imbalance, and rerun on DeepDish5: **13 tests, 0 failures, 0 errors**. The tests cover exact aggregation, one-step centralized equivalence, tiny/empty splits, sampling, bad updates, broadcast isolation/fresh optimizer repeatability, empty client handling, sample-weighted metrics, invalid pixels/labels/C, restart identity, empty input lists, a 1:99,999 aggregation-count ratio, and explicit nonfinite local-loss rejection. The empty-list loader correction does not change the nonempty supplied data path or any completed main result. Scientific evidence below extends beyond these unit checks.

### A_exact_aggregation

**Hypothesis:** Unequal training counts produce the exact weighted mean

**Setup:** Three deterministic float32 vectors; n=[1,3,6]

**Expected behavior / criterion:** Maximum error <= 1e-6

**Observed outcome:** PASS.

```json
{
  "max_abs_error": 2.384185791015625e-07,
  "tolerance": 1e-06,
  "actual": [
    -0.5,
    2.8000001907348633,
    5.400000095367432
  ],
  "expected": [
    -0.5,
    2.799999952316284,
    5.400000095367432
  ]
}
```

**Interpretation:** Direct numerical verification of sample weighting

**Limitations:** Small deterministic tensors; end-to-end behavior tested separately

### B_centralized_equivalence

**Hypothesis:** One local full-batch step equals centralized full-batch SGD

**Setup:** 17 synthetic examples split 2/5/10; all participate; E=1; LR=.03; same MLP

**Expected behavior / criterion:** Max parameter error <= 3e-7

**Observed outcome:** PASS.

```json
{
  "max_abs_error": 7.450580596923828e-09,
  "tolerance": 3e-07,
  "client_sizes": [
    2,
    5,
    10
  ],
  "lr": 0.03,
  "explanation": "One full-batch local step; mean loss gradients weighted by training counts"
}
```

**Interpretation:** Validates the gradient/parameter averaging identity

**Limitations:** Identity applies to one full-batch step, not multiple local mini-batches

### C_sampling

**Hypothesis:** Each eligible client has probability C each round

**Setup:** 10,000 draws for each C; duplicate seeded stream

**Expected behavior / criterion:** Exact k, no duplicates, identical seeded sequences; all client counts within 5 binomial SD

**Observed outcome:** PASS.

```json
[
  {
    "fraction": 0.02,
    "rounds": 10000,
    "expected_per_client": 200.0,
    "binomial_sd": 14.0,
    "min": 166,
    "max": 230,
    "max_abs_z": 2.4285714285714284,
    "correct_unique_reproducible": true
  },
  {
    "fraction": 0.05,
    "rounds": 10000,
    "expected_per_client": 500.0,
    "binomial_sd": 21.79449471770337,
    "min": 432,
    "max": 574,
    "max_abs_z": 3.395352861284314,
    "correct_unique_reproducible": true
  },
  {
    "fraction": 0.1,
    "rounds": 10000,
    "expected_per_client": 1000.0,
    "binomial_sd": 30.0,
    "min": 924,
    "max": 1069,
    "max_abs_z": 2.533333333333333,
    "correct_unique_reproducible": true
  }
]
```

**Interpretation:** Participation matches expected marginal random variation

**Limitations:** Counts across clients are dependent within a round; diagnostic is not proof of uniformity

### D_controlled_learning

**Hypothesis:** SGD learns a spatial signal; extremely small LR learns slower

**Setup:** 20 synthetic clients; 960 train / 320 validation; balanced two-class signal; same initial weights and selected clients

**Expected behavior / criterion:** LR .1: final validation >90%, loss <50% of initial; >20 percentage points above LR .0001

**Observed outcome:** PASS.

```json
{
  "initial_train_loss": 4.051352183024089,
  "final_train_loss": 0.0025787144433706997,
  "suitable_lr_accuracy": 1.0,
  "tiny_lr_accuracy": 0.0125,
  "suitable_lr": 0.1,
  "tiny_lr": 0.0001
}
```

**Interpretation:** Checks useful learning and a controlled optimization response

**Limitations:** Synthetic binary task is easier than 62-class handwriting

### E_label_randomization

**Hypothesis:** Destroying training-label information prevents meaningful validation learning

**Setup:** Full development pool, 40 rounds, C=.1 E=3, matched init/sampling/batches/LR; permute labels within each client's training set; validation unchanged

**Expected behavior / criterion:** Correct-label final validation exceeds shuffled-label validation by >15 percentage points

**Observed outcome:** PASS.

```json
{
  "correct_validation": 0.66364913375995,
  "shuffled_validation": 0.07647885125643827,
  "gap": 0.5871702825035118
}
```

**Interpretation:** Within-client permutation preserves label counts and client sizes while breaking image-label correspondence

**Limitations:** One matched seed; finite shuffled data can retain accidental correlations; does not prove privacy

### F_heterogeneity

**Hypothesis:** Stronger label segregation creates client drift under several local SGD steps

**Setup:** Reassign only the 25,418 TRAINING examples into original size vector; shuffled vs label-sorted assignment; unchanged validation; 40 rounds C=.1 E=3

**Expected behavior / criterion:** Sorted partition has lower entropy; examine validation gap rather than impose a universal ordering

**Observed outcome:** PASS.

```json
{
  "original": {
    "mean_entropy_bits": 5.111069115069352,
    "min_entropy_bits": 4.682227515231736,
    "max_entropy_bits": 5.272160636213872
  },
  "controlled": [
    {
      "partition": "randomized_homogeneous",
      "mean_entropy_bits": 5.183606582265064,
      "min_entropy_bits": 4.949796128031286,
      "max_entropy_bits": 5.346731548001385,
      "final_validation_accuracy": 0.6987669736226003,
      "best_validation_accuracy": 0.6987669736226003,
      "train_samples": 25418,
      "cumulative_examples_seen": 306447
    },
    {
      "partition": "label_sorted_heterogeneous",
      "mean_entropy_bits": 0.3780192565234542,
      "min_entropy_bits": -0.0,
      "max_entropy_bits": 2.2846815292683766,
      "final_validation_accuracy": 0.2781332917121898,
      "best_validation_accuracy": 0.2923365069455283,
      "train_samples": 25418,
      "cumulative_examples_seen": 306447
    }
  ],
  "homogeneous_minus_heterogeneous": 0.4206336819104105
}
```

**Interpretation:** This controlled intervention isolates partition structure at matched sizes, counts, rounds and sampling; sample exposure is equal

**Limitations:** Extreme artificial partitions; original user partitions remain required main results; label entropy does not measure writer-style heterogeneity

### edge_excessive_lr

**Hypothesis:** Excessive LR destabilizes learning and nonfinite values are rejected

**Setup:** Same synthetic signal; LR=1e6; 5 rounds E=3

**Expected behavior / criterion:** Clear deterioration or explicit FloatingPointError

**Observed outcome:** PASS.

```json
{
  "nonfinite_rejected": false,
  "final_train_loss": 155043.08697916666,
  "suitable_lr_final_train_loss": 0.0025787144433706997
}
```

**Interpretation:** No failed update is substituted or silently included in averaging

**Limitations:** Synthetic stress case, not an observed source-data failure

### G_reproducibility

**Hypothesis:** Leading settings should retain useful validation performance across seeds

**Setup:** Top two primary-seed settings repeated with seeds 2027,2028; split seed stays 490

**Expected behavior / criterion:** Identical initialization within seed; quantify cross-seed variation

**Observed outcome:** PASS.

```json
{
  "identical_primary_initialization": true,
  "repeats": [
    {
      "fraction": 0.05,
      "epochs": 3,
      "seeds": [
        2026,
        2027,
        2028
      ],
      "best_mean": 0.7193694396753552,
      "best_sd": 0.009892275247379949,
      "final_mean": 0.7151552988918372,
      "final_sd": 0.011444986750628149,
      "best_min": 0.7101607616669268,
      "best_max": 0.7298267519900109
    },
    {
      "fraction": 0.1,
      "epochs": 3,
      "seeds": [
        2026,
        2027,
        2028
      ],
      "best_mean": 0.727745694812965,
      "best_sd": 0.002263598139306682,
      "final_mean": 0.7239997918942823,
      "final_sd": 0.005654150393930831,
      "best_min": 0.7254565319182145,
      "best_max": 0.7299828312782893
    }
  ]
}
```

**Interpretation:** Reports sample SD across three seeds instead of treating one trajectory as definitive

**Limitations:** Three seeds are insufficient for narrow population uncertainty; seed changes init, selection and minibatch order

### F_client_imbalance_observational

**Hypothesis:** Client size and heterogeneity may accompany uneven benefits

**Setup:** 100 original clients; baseline final-round validation minus round-zero validation

**Expected behavior / criterion:** Describe correlations and extremes, without causal claims

**Observed outcome:** PASS.

```json
{
  "correlations": {
    "participation": -0.09366345824934486,
    "train_n": 0.28947642287540737,
    "entropy_bits": -0.06802676471819066
  },
  "quartile_groups": {
    "train_n": {
      "lowest_quartile_mean_benefit": 0.578664694810019,
      "highest_quartile_mean_benefit": 0.6525979960806545
    },
    "entropy_bits": {
      "lowest_quartile_mean_benefit": 0.6018690408253591,
      "highest_quartile_mean_benefit": 0.5916288389641524
    }
  },
  "least_benefit": [
    {
      "client_id": 29,
      "participation": 5,
      "train_n": 268,
      "entropy_bits": 5.187157904098231,
      "observed_classes": 50,
      "initial_accuracy": 0.04411764705882353,
      "final_accuracy": 0.45588235294117646,
      "benefit": 0.4117647058823529
    },
    {
      "client_id": 5,
      "participation": 7,
      "train_n": 306,
      "entropy_bits": 4.9879923107644295,
      "observed_classes": 48,
      "initial_accuracy": 0.0,
      "final_accuracy": 0.4155844155844156,
      "benefit": 0.4155844155844156
    },
    {
      "client_id": 25,
      "participation": 10,
      "train_n": 94,
      "entropy_bits": 4.802129908483114,
      "observed_classes": 38,
      "initial_accuracy": 0.0,
      "final_accuracy": 0.4166666666666667,
      "benefit": 0.4166666666666667
    }
  ],
  "most_benefit": [
    {
      "client_id": 4,
      "participation": 6,
      "train_n": 300,
      "entropy_bits": 5.062819324426839,
      "observed_classes": 53,
      "initial_accuracy": 0.05263157894736842,
      "final_accuracy": 0.868421052631579,
      "benefit": 0.8157894736842106
    },
    {
      "client_id": 62,
      "participation": 9,
      "train_n": 307,
      "entropy_bits": 5.087221028663311,
      "observed_classes": 53,
      "initial_accuracy": 0.012987012987012988,
      "final_accuracy": 0.8441558441558441,
      "benefit": 0.8311688311688311
    },
    {
      "client_id": 6,
      "participation": 8,
      "train_n": 272,
      "entropy_bits": 5.13505737112562,
      "observed_classes": 52,
      "initial_accuracy": 0.0,
      "final_accuracy": 0.8823529411764706,
      "benefit": 0.8823529411764706
    }
  ],
  "participation_min": 2,
  "participation_max": 13,
  "expected_participation": 8.0
}
```

**Interpretation:** Local validation sets differ in size and class mix; sampling frequency is a random exposure measure

**Limitations:** Observational associations are not causal; small client validation sets have substantial binomial noise

## Representative clients and data heterogeneity

Five clients were selected systematically from training-only summaries. Ties use smaller client IDs and selected IDs are unique. The typical client minimizes standardized squared distance to medians of training size, observed-class count and entropy. Entropy is Shannon entropy in bits; all counts in this table are post-split training counts.

| Client | Criterion | Training examples | Observed classes | Dominant class fraction | Entropy (bits) |
|---:|---|---:|---:|---:|---:|
| 67 | largest training set | 314 | 54 | 0.0860 | 5.0836 |
| 25 | smallest training set | 94 | 38 | 0.1170 | 4.8021 |
| 16 | most observed training classes | 300 | 57 | 0.0767 | 5.2205 |
| 36 | fewest observed training classes | 182 | 37 | 0.0604 | 4.9641 |
| 86 | closest to median size, diversity and entropy | 260 | 51 | 0.1038 | 5.1149 |

Development counts range from 56 for ID 45 to 1837 for ID 24; the largest/smallest frequency ratio is 32.80. Original clients' training class counts range from 37 to 57; training entropy ranges from 4.6822 to 5.2722 bits. All clients lack at least some classes. Size differences and writer identity provide heterogeneity even where class entropies are similar. Overall class imbalance must not be confused with between-client heterogeneity.

## Figures and non-standard diagnostic

- `overall_class_distribution.png` / `.pdf`: Pre-split development pool only; no held-out test labels.
- `five_client_distributions.png` / `.pdf`: Systematic selection by size extremes, diversity extremes, and median profile.
- `client_class_heatmap.png` / `.pdf`: Row-normalized class frequencies retain original user boundaries.
- `baseline_convergence.png` / `.pdf`: Entire aggregate train/validation partitions after every updated global model; includes round zero.
- `hyperparameter_accuracy.png` / `.pdf`: Six controlled configurations, common split/initialization/LR/batch/80-round budget.
- `accuracy_vs_client_epochs.png` / `.pdf`: Equal rounds have different computation: 160 to 2400 client-epochs.
- `seed_stability.png` / `.pdf`: Pointwise sample SD across seeds; not a confidence interval.
- `participation_heterogeneity_benefit.png` / `.pdf`: Baseline original clients: final round minus round zero; observational associations only.
- `participation_histogram.png` / `.pdf`: Actual baseline selections, 800 total client participations.
- `scientific_controls.png` / `.pdf`: Matched controlled interventions; development/synthetic data only.
- `pilot_learning_rates.png` / `.pdf`: Shared LR selected by mean round-12 validation across E=1 and E=3.

The required non-standard plot is **Client Participation, Heterogeneity, and Learning Benefit**. X is baseline participation count, Y is final-minus-initial client validation accuracy in percentage points, color is training-label entropy, and point area represents training size. It includes a zero-benefit line, the expected count 8 line, colorbar, size legend, and labels on benefit extremes.

Actual participation spans 2 to 13 rounds. Pearson correlations of benefit with participation, size and entropy are -0.094, 0.289, -0.068. Low-benefit IDs are [29, 5, 25]; high-benefit IDs are [4, 62, 6]. These points describe uneven client outcomes, not proven effects of participation. Local validation sample size, starting accuracy, class mix and writer difficulty can explain extremes; the saved per-client CSV supports follow-up. The quartile summaries above quantify small/large and low/high entropy groups without causal claims.

## Frozen selection and single held-out evaluation

Among top two primary-seed settings, maximize mean best validation over seeds 2026,2027,2028; then choose seed/checkpoint with highest validation accuracy (loss breaks checkpoint ties).

Chosen C=0.10, E=3, LR=0.3, batch=32, seed=2026, checkpoint round=80. Best selected validation accuracy: **73.00%**. The complete selection and checkpoint SHA-256 were saved before opening test data.

The held-out file is a NumPy object array of shape [1] with one aggregated dictionary, images shape [3621, 28, 28], source labels dtype int32. It contains **3,621 test examples**. The frozen checkpoint obtains **74.12% test accuracy (2,684/3,621 correct)** and test CE 0.986716. This is one aggregate test result; per-writer test attribution is unavailable. Test results were not used to revise the configuration, number of rounds, normalization, checkpoint or seed. The result is cached against the selection hash to prevent accidental rescreening.

The exactly matched Part 2 zero-noise evaluations add a fixed-round-80 summary for this configuration: seed-2026/2027/2028 test accuracies are 74.12%, 72.88%, and 73.13%, yielding 73.38% +/- 0.66 percentage points (sample SD). All use the same 3,621 held-out examples; the official selected-checkpoint result remains 74.12%.

## Limitations

- A fixed 80-round budget and this small MLP do not establish best attainable EMNIST performance. Rare classes and ambiguous uppercase/lowercase characters can remain difficult; numeric IDs alone limit semantic error analysis.
- Three seeds only for the top two configurations provide limited uncertainty evidence. Validation-based selection over settings, checkpoints and seeds has selection optimism. The held-out result is the independent final check; do not tune further against it.
- The test set is aggregated, so sample-wise accuracy cannot establish writer-level fairness, out-of-client generalization, or client-specific test uncertainty. Label entropy measures only one dimension of non-IID data.
- Artificial homogeneous/heterogeneous partitions are controlled interventions, not replacements for real client boundaries. The label-randomization experiment is a learning sanity check, not a statistical privacy guarantee.
- This is sequential synchronous simulation on one GPU. Measured loop time includes evaluation/I/O and is not a distributed latency or energy benchmark. Other server activity can affect runtime. Exact bitwise reproduction across hardware or PyTorch releases is not guaranteed.

## Proposed three-month extended validation plan

1. **Month 1: reliability and uncertainty.** Freeze this final test evaluation. Add 10-20 prespecified seeds for all six settings, bootstrap validation at the client level, collect rare-class recall/confusion and calibration metrics using a verified character mapping. Repeat preprocessing and LR ablations on development data with fixed search budgets. Validate data lineage, resumption after process termination and CPU/CUDA numerical agreement.
2. **Month 2: heterogeneity and robustness.** Run matched-size Dirichlet label partitions over a prespecified concentration grid, writer-style/quantity skew experiments, controlled missing classes, singleton and highly unequal clients, client drift norms, and sample-versus-client fairness summaries. Evaluate dropout, stragglers and corruption with explicit failure policies. Compare fixed-round, fixed-client-epoch and fixed-example-exposure budgets; keep original writer partitions as the principal setting.
3. **Month 3: confirmatory generalization and systems.** Reserve a newly held-out writer-level development split for confirmatory protocol selection, pre-register comparisons, then obtain a fresh untouched final test source before any new final scoring. Measure repeated hardware runtime/memory/energy and communication proxies with confidence intervals. Analyze accuracy/fairness tradeoffs, document failure cases, audit all plots against source metrics, and write a reproducible artifact package. Related Ray and privacy extensions are discussed in the completed component notes.

Publication note: personal paths and host addresses in these Markdown excerpts are omitted. Original numerical JSON/CSV evidence is unchanged. Generated notes can restore machine metadata; consult the component README before regeneration or publication.
