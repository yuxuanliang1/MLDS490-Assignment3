# MLDS 490 Assignment 3 — Federated Learning and Differential Privacy

**FedAvg on Federated EMNIST, Ray Actor Parallelization, and Laplace Input Perturbation**

Northwestern University · MLDS 490 · Assignment 3  
Author: **Yuxuan Liang**  
Repository: https://github.com/yuxuanliang1/MLDS490-Assignment3

## Overview

- **Part 1.1 — Sequential FedAvg:** uniform client sampling, sample-count-weighted aggregation, client-fraction/local-epoch comparisons, and scientific correctness controls.
- **Part 1.2 — Ray Parallel FedAvg:** four-client synchronous updates using Ray Core Actors, with matched numerical equivalence, concurrency, lifecycle, and performance measurements.
- **Part 2 — Laplace Input Perturbation:** fixed independent pixelwise noise on local training images, evaluated through a seven-scale utility comparison and explicit privacy limitations.

All parts use a `784 → 128 → ReLU → 62` fully connected network with 108,478 parameters, mean cross-entropy, and plain SGD. The fixed preprocessing is `1-X`. Client-local data boundaries are simulated on shared hardware.

## Main results

| Component | Key result |
|---|---|
| Sequential FedAvg | **74.12%** held-out test accuracy |
| Ray Parallel FedAvg | **73.79%** held-out test accuracy |
| Sequential/Ray equivalence | Maximum matched parameter difference: **0.0** |
| Ray execution time | **276.14 s** versus **11.95 s** sequential; approximately **23.1× slower** |
| Laplace perturbation, b=0.05 | **72.66%** mean round-80 held-out test accuracy |
| Noise-scale experiments | **7 scales × 3 seeds = 21 training runs** |

These are different evaluation protocols, not a controlled ranking: Part 1.1 selects seed 2026/round 80 with ten clients per round; Ray uses four clients and selects round 78. Part 2 reports fixed round-80 three-seed means; its zero-noise control also gives **73.38% ± 0.66 percentage points** for the selected Part 1.1 configuration on the same test set.

## Repository structure

```text
MLDS490-Assignment3/
├── README.md
├── Assignment3-data/          # Local course data; obtain separately
├── references/               # Local course materials; check redistribution rights
├── part1_1/                  # Sequential algorithm, tests, and workflows
├── part1_2/                  # Ray Actors and systems validation
├── part2/                    # Laplace mechanism and experiment workflows
├── part1_1_deepdish5_outputs/ # Final Part 1.1 evidence
├── part1_2_outputs/           # Ray evidence; clients/*.npz are private derived data
├── part2_outputs/            # Noise-grid evidence and frozen selection
├── part1_1_outputs/          # Local exploratory archive; not final evidence
├── report/                   # Self-contained Overleaf source and figures
└── tmp/                      # Local review intermediates; not for publication
```

Detailed guides: [Part 1.1](part1_1/README.md), [Part 1.2](part1_2/README.md), [Part 2](part2/README.md), and [Overleaf](report/README.md). Output indexes identify the existing result files; publication should include a reviewed subset rather than entire local archives.

## Requirements

The completed runs used Linux on Northwestern DeepDish5. These are recorded versions, not claims that other versions reproduce bitwise results:

| Dependency | Tested version / role |
|---|---|
| Python | 3.12.6 |
| PyTorch | 2.14.0+cu130 |
| CUDA build | 13.0; compatible GPU/driver needed for GPU runs |
| NumPy | 2.5.3 |
| Matplotlib | 3.11.2 |
| psutil | 7.2.2 |
| Ray | 2.60.0; Part 1.2 only |

Use an isolated Python environment with these libraries and a PyTorch build compatible with the chosen hardware. No dedicated dependency lockfile is supplied; the existing component `package_versions.txt` files record the full historical environments. The experiments used an RTX 5060 Ti with compute capability 12.0, 48 logical CPUs, and 93.8 GiB RAM. Parts 1.1 and 2 support `--device cpu`; the selected Ray run uses four CPU Actors, although its validation also attempts a GPU strategy. Bash examples below assume a Linux shell and run from the repository root.

## Dataset setup

Obtain the instructor-provided files separately and place `train_data.npy` and `test_data.npy` in [Assignment3-data/](docs/DATASET.md). Nothing downloads the data automatically. Redistribution permission has not been established; keep the raw files and Ray-generated client `.npz` files out of the public repository.

The development partition has **100 clients, 62 classes, and 31,825 examples**. Client-specific 80/20 splits give **25,418 training** and **6,407 validation** examples. The separate held-out partition has **3,621 examples**. Reuse [the saved split manifest](part1_1_deepdish5_outputs/splits.json), with split seed 490, for matched reproduction. The loader accepts trusted NumPy object arrays; use the course-supplied files.

## Running the experiments

The assignment is complete. The following commands document reproduction; they are not needed to read the report. Use a separate working copy for new runs: several commands write logs, summaries, or generated documentation, and cache reuse is not a fresh experiment. For a fresh reproduction, retain the input data and saved split manifest but create new output trees; do not carry historical frozen selections or preservation snapshots into them.

### Quick checks and inspection

```bash
export MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m part1_1 inspect --device cpu
python -m unittest discover -s part1_1/tests -v
```

The unit suite uses small synthetic fixtures. It is distinct from the full scientific controls below, which include training and can take substantially longer.

### Full reproduction, in order

The GPU examples require an allocated compatible device. Complete Part 1.1 before Part 2's full zero-noise validation; finish Part 1.2 before Part 2 initializes its preservation snapshot.

```bash
# Part 1.1: pilot, comparison grid, controls, selection, then held-out scoring
python -m part1_1 pilot --device cuda:0
python -m part1_1 suite --device cuda:0
python -m part1_1 validate --device cuda:0
python -m part1_1 select --device cuda:0
python -m part1_1 test --device cuda:0
python -m part1_1 plots

# Part 1.2: matched numerical checks and CPU/GPU benchmarks, then full runs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m part1_2.workflow prepare
python -m part1_2.workflow validate
python -m part1_2.extra_validation
python -m part1_2.workflow train
python -m part1_2.workflow artifacts

# Part 2: mechanism checks, pilots, 21-run grid, controls, frozen test protocol
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m part2.workflow init --device cuda:0
python -m part2.workflow math --device cuda:0
python -m part2.workflow pilot --device cuda:0
python -m part2.workflow grid --device cuda:0
python -m part2.workflow science --device cuda:0
python -m part2.audit --pretest
python -m part2.workflow freeze
python -m part2.workflow test --device cuda:0
python -m part2.workflow artifacts
```

Ray's `train` phase includes validation-based selection and its held-out evaluation; it is not a training-only command. Part 2's `math` phase includes a short zero-noise training control, not just random-number statistics. See the component guides for configurable runs, final audits, caching, and failure recovery.

### Regenerate plots from completed outputs

```bash
python -m part1_1 plots
python -m part1_2.workflow artifacts
python -m part2.workflow artifacts
python report/tools/regenerate_figures.py
```

These commands do not train models or score the test set. Part 2's image panel also reads development images. The last command regenerates nine report-sized PNGs; the others produce component figures, and some also refresh generated notes. Review generated documentation for machine metadata before publication.

## Scientific validation

| Question | Representative evidence | Details |
|---|---|---|
| Correct FedAvg arithmetic? | Aggregation error ≈2.38e-7; one-step centralized-SGD error ≈7.45e-9 | [Part 1.1 validation](part1_1_deepdish5_outputs/validation/results.json) |
| Does learning depend on task structure? | Correct/randomized labels: 66.36%/7.65%; homogeneous/segregated partitions: 69.88%/27.81% | [Part 1.1 notes](part1_1_deepdish5_outputs/REPORT_NOTES.md) |
| Does Ray preserve learning and execute concurrently? | Matched trajectory difference 0.0; four overlapping Actor intervals in all 80 rounds | [Trajectory comparison](part1_2_outputs/trajectory_equivalence.json), [lifecycle measurements](part1_2_outputs/lifecycle_concurrency_audit.json) |
| Is perturbation implemented correctly? | Million-sample moments/quantiles/tails, pixel correlations, exact zero-noise recovery, non-IID and extreme-noise tests | [Part 2 validation](part2_outputs/validation/results.json) |

The non-standard plots examine client participation/heterogeneity/benefit, Actor execution intervals, and client-level Laplace robustness. [Final noise results](part2_outputs/final_results.csv) retain all 21 runs, including degraded outcomes.

## Report

The academic report is under [report/](report/README.md), with [main.tex](report/main.tex) as its Overleaf entry point. It contains the required plots, scientific controls, edge cases, privacy interpretation, reproduction instructions, and proposed three-month validation plan. Upload the current folder contents to Overleaf. The final [31-page compiled report](MLDS_490_Assignment_3.pdf) is included alongside the Overleaf sources.

## Reproducibility and limitations

The original runs used DeepDish5; shared-host timings and exact numerical replay are environment-specific. The shell launchers retain historical server settings and are not universally portable; use the documented module commands. The small workload makes Actor startup dominant despite actual concurrency.

The predefined choice **b=0.05** limits mean final validation-accuracy degradation to at most two percentage points: 72.40% becomes 71.80%, a 0.60-point decrease. This is a practical utility tolerance, not a significance threshold or privacy standard. Fixed input noise, public reproducibility keys, unchanged labels, and clean-data metric releases do **not** establish a verified end-to-end epsilon-DP guarantee.

Historical source/preservation hashes include documentation. Documentation revisions can therefore cause an old full-tree preservation audit to fail even when experimental code, data, and results are unchanged. Keep historical snapshots intact; a fresh reproduction creates its own snapshot after prerequisite work is complete.

### Publication scope

Publish reviewed code, documentation, report sources, and selected numerical evidence. Keep raw `Assignment3-data/*.npy`, `part1_2_outputs/clients/*.npz`, local environments, `tmp/`, exploratory outputs, transfer/completion archives, and unreviewed machine logs private. Checkpoint files and source/preservation/environment records also need deliberate review; they are not necessary to compile the report. Course PDFs and example-image figures require appropriate redistribution permission. The `.gitignore` and selective release exclude local data, machine records, archives, and redundant checkpoints. Sanitized historical records are in [public_evidence/](public_evidence/README.md). No raw dataset is embedded in these documents.

## References

- McMahan et al., [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html), AISTATS 2017.
- Caldas et al., [LEAF: A Benchmark for Federated Settings](https://arxiv.org/abs/1812.01097), 2018.
- Ray Project, [Ray Core Actors](https://docs.ray.io/en/latest/ray-core/actors.html).
- Additional research and course references are in [the report bibliography](report/references.bib).
