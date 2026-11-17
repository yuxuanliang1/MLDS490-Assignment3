# Part 1.1: Sequential FedAvg

[Project overview](../README.md) | [Final evidence index](../part1_1_deepdish5_outputs/INDEX.md) | [Report](../report/README.md)

## Purpose and implementation

This completed component implements synchronous FedAvg without a specialized federated-learning library. Each selected client starts from the same global parameters, runs fresh local SGD, and returns parameters weighted by its training count. Uniform sampling without replacement chooses 2, 5, or 10 of 100 clients for C=0.02, 0.05, or 0.10. The shared model is Flatten -> Linear(784,128) -> ReLU -> Linear(128,62), with 108,478 parameters and raw logits.

| Module | Responsibility |
|---|---|
| `data.py`, `model.py` | Trusted input loading, seeded splits, class analysis, network |
| `federation.py` | Client SGD, sampling, aggregation, sample-weighted evaluation |
| `runner.py` | Configuration, checkpoints, cached runs, round-boundary recovery |
| `workflow.py`, `cli.py`, `__main__.py` | Pilot, grid, validation, selection, test, plotting commands |
| `science.py`, `tests/test_correctness.py` | Scientific controls and correctness fixtures |
| `plotting.py`, `documentation.py` | Figures and generated numerical notes |
| `audit.py`, `provenance.py`, `probe.py` | Artifact checks, provenance, environment/device probe |

## Dependencies and input

Use Python, PyTorch, NumPy, Matplotlib, and psutil; Ray is unnecessary for this component. The [root requirements](../README.md#requirements) give the recorded versions and hardware. Parts 1.1 and 2 used two CPU threads with a compatible CUDA device; `--device cpu` is also supported.

Place the trusted instructor files in `Assignment3-data/` as described in the [dataset guide](../docs/DATASET.md). The 31,825 development examples are split independently within each client using seed 490: 25,418 train and 6,407 validation. The 3,621 test examples remain separate. Inputs use `1-X`; labels remain numeric IDs 0-61. Reuse `part1_1_deepdish5_outputs/splits.json` for matched experiments.

## Running from the repository root

Use a separate working copy for reproduction so completed artifacts remain intact. These commands write outputs; none is needed merely to inspect saved results. Bash examples assume Linux and an already activated environment:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m part1_1 inspect --device cpu
python -m part1_1 --help
```

`inspect` creates/reuses the split manifest and writes class-count analysis. For a small CPU correctness suite without rewriting the saved test log:

```bash
python -m unittest discover -s part1_1/tests -v
```

The workflow wrapper below executes the same suite and writes its evidence under `validation/`. Full reproduction uses the following order; substitute `--device cpu` for GPU commands when appropriate:

```bash
python -m part1_1.run_tests
python -m part1_1 inspect --device cuda:0
python -m part1_1 pilot --device cuda:0
python -m part1_1 suite --device cuda:0
python -m part1_1 validate --device cuda:0
python -m part1_1.audit
python -m part1_1 select --device cuda:0
python -m part1_1 test --device cuda:0
python -m part1_1 plots
python -m part1_1 report
python -m part1_1.audit --final
```

The pilot compares learning rates 0.01, 0.1, and 0.3 over 12 rounds at E=1,3; the shared selected rate is 0.3. `suite` runs six 80-round C/E settings and repeats the two leading settings at seeds 2027 and 2028. `validate` includes training-based scientific controls. `select` freezes the validation-based choice before `test` accesses held-out examples. Cached test scores are reused rather than used to retune settings.

A separate configurable run can reuse the split with a different output directory:

```bash
mkdir -p new_outputs
cp part1_1_deepdish5_outputs/splits.json new_outputs/splits.json
python -m part1_1 train --device cpu --rounds 80 \
  --fraction 0.10 --epochs 3 --lr 0.3 --batch-size 32 \
  --seed 2026 --threads 2 --name controlled_run --output-dir new_outputs
```

The default CLI output is `part1_1_deepdish5_outputs/`; `--data-dir` and `--output-dir` are supported. The test wrapper and artifact audit use their own fixed/default output locations, so a custom training directory is not a drop-in replacement for the complete multi-part workflow. Completed run configurations are cached; incomplete runs restore the last committed round. Preserve finalized outputs rather than deleting them to force reruns.

## Scientific validation and results

The 13-test suite checks weighted metrics/aggregation, the one-step centralized-SGD identity, tiny/empty clients, missing classes, bad counts/tensors, seeded sampling, broadcast isolation, and restart equivalence. Scientific `validate` adds learnable signals, randomized labels, controlled partition heterogeneity, seed variability, and excessive-learning-rate stress.

- Unequal-count aggregation error: approximately **2.38e-7**; centralized-SGD identity error: **7.45e-9**.
- Correct versus randomized labels: **66.36% versus 7.65%** validation accuracy.
- Homogeneous versus label-segregated partitions: **69.88% versus 27.81%**.
- Selected configuration: **C=0.10, E=3, learning rate 0.3, batch size 32**.
- Validation-selected seed 2026/round 80: **74.12%** test accuracy (2,684/3,621).
- Fixed round-80 seeds 2026-2028: **73.38% +/- 0.66 percentage points**, supported by exactly matched Part 2 zero-noise trajectories on the same test set. This does not replace the selected-checkpoint result.

See [validation/results.json](../part1_1_deepdish5_outputs/validation/results.json), [comparison.csv](../part1_1_deepdish5_outputs/comparison.csv), and [final_test.json](../part1_1_deepdish5_outputs/final_test.json). Accuracy alone is not the correctness argument; each control has an independent expected response.

## Outputs and plots

Each run saves configuration, round-zero-through-80 history, selected client IDs/counts, per-client metrics, initial/best/last checkpoints, and a summary. Best checkpoints maximize validation accuracy, then minimize validation loss, then prefer the earlier round. `python -m part1_1 plots` regenerates figures from completed saved outputs without training or test scoring. `python -m part1_1 report` regenerates notes and may restore historical wording; review generated Markdown before publication.

The [output index](../part1_1_deepdish5_outputs/INDEX.md) includes class distributions, convergence, hyperparameter comparisons, and the non-standard participation/heterogeneity/benefit plot. `part1_1_outputs/` is a separate exploratory archive, not the final evidence source.

## Platform and publication notes

`run_server.sh` and `finalize_server.sh` retain DeepDish5-specific Python/GPU defaults. The first supports `FEDAVG_PYTHON` and `FEDAVG_PHYSICAL_GPU`; the finalizer fixes physical GPU 0 and writes environment records. Use the module commands on other hosts instead of assuming the shell launchers are portable. No SSH account or personal directory is needed in these instructions.

Raw input files, checkpoints, machine logs, and historical source manifests require publication review. Documentation edits do not change experiments, but historical full-tree hashes may include the old README bytes. The later Ray and Laplace components and the combined report are complete; follow their linked guides for those workflows.
