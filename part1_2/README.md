# Part 1.2: Parallel FedAvg with Ray Core Actors

[Project overview](../README.md) | [Evidence index](../part1_2_outputs/INDEX.md) | [Report](../report/README.md)

## Purpose and structure

This completed component executes synchronous FedAvg with exactly four clients per round. `core.py` implements local-training Actors, deterministic aggregation order, timeout handling, and cleanup using the Part 1.1 model and client routines. `workflow.py` prepares client files, compares numerical results, benchmarks resources, and runs matched sequential/parallel trajectories. `extra_validation.py` adds bounded failure/replay checks; `artifacts.py` produces plots and notes; `audit.py` checks completed artifacts.

Each round creates only the four selected Actors, waits for all valid updates, aggregates in sampled order, then kills the Actors and verifies process exit and CPU return. CPU Actors reserve one CPU and zero GPUs each. The alternative benchmark reserves one CPU and 0.25 GPU per Actor on one device. The selected run uses CPU Actors, E=3, learning rate 0.3, batch size 32, seed 2026, and 80 rounds. Simulated data ownership does not provide filesystem or network security.

## Requirements and input

Use the [recorded Python/PyTorch environment](../README.md#requirements), plus **Ray 2.60.0** and psutil. The completed run used Linux on DeepDish5 and one thread per Actor and driver. CPU execution needs four available logical CPU resources. Validation also attempts fractional-GPU execution; a compatible allocated CUDA device is needed to reproduce the successful GPU benchmark. The code records GPU failure and can select a valid CPU strategy; timings will then differ from the recorded benchmark.

Provide both instructor files under `Assignment3-data/` and reuse `part1_1_deepdish5_outputs/splits.json`. Run Part 1.1 `inspect` first if setting up a fresh reproduction, while preserving the supplied manifest. `prepare` writes per-client training arrays to `part1_2_outputs/clients/*.npz`; these are dataset-derived files and must not be publicly committed without permission.

## Reproduction order

Run from the repository root in a separate working copy, with the selected device allocated through the host's usual procedure:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m part1_2.workflow prepare
python -m part1_2.workflow validate
python -m part1_2.extra_validation
python -m part1_2.workflow train
python -m part1_2.workflow artifacts
python -m part1_2.audit
```

`validate` performs matched one-round comparisons, failure injection, sampling checks, and three-repeat CPU/GPU/sequential benchmarks. It writes the execution choice used by `train`. `extra_validation` uses CPU Actors for tiny/missing-class clients and deterministic replay, with an intentionally unschedulable GPU request to test timeout cleanup. These are scientific computations, not read-only checks.

`train` runs a five-round pilot and matched 80-round sequential/Ray trajectories, freezes the best-validation checkpoint, and **also evaluates held-out test data**. There is no separate Ray test CLI phase and no `--device` option on this workflow. `artifacts` regenerates figures, lifecycle measurements, and notes from saved outputs without training or test scoring. `audit` reads checkpoints/results and writes its audit/index.

Output locations are fixed relative to the current working directory. Completed validation/benchmarks and runs are cached; incomplete runs resume from the atomic last checkpoint. Use a separate fresh output tree for a new scientific repetition rather than deleting or overwriting the finalized evidence. Partial updates never enter an aggregate, and deterministic replay starts from the last committed round.

## Results and validation

- Matched CPU local/global updates and full trajectories agree exactly: maximum parameter/metric difference **0.0**.
- Four distinct Actor training intervals overlap in **all 80 rounds**, with process exit and CPU-resource return verified each round.
- Validation selects **round 78**; held-out accuracy is **73.79%** (2,672/3,621).
- Full workload: **276.14 s Ray versus 11.95 s sequential**, speedup 0.0433x, or approximately **23.1x slower**. Actor startup dominates the small model; concurrency does not imply speedup.
- Injected exceptions, allocation errors, non-finite updates, unavailable GPU resources, interruption, and replay exercise defined failure paths without deliberately exhausting physical memory.

Sources: [validation.json](../public_evidence/part1_2_outputs/validation.json), [extra_validation.json](../part1_2_outputs/extra_validation.json), [trajectory_equivalence.json](../part1_2_outputs/trajectory_equivalence.json), [test.json](../part1_2_outputs/test.json), and [performance_summary.csv](../part1_2_outputs/performance_summary.csv). Part 1.1's ten-client test result uses a different protocol and is not a controlled Ray comparison.

## Output structure and visualization

`parallel/` and `sequential/` contain histories, selections, timings, checkpoints, and summaries. Actor events include client IDs, PIDs, and interval timestamps. The [output index](../part1_2_outputs/INDEX.md) links numerical evidence and five PNG/PDF figure pairs: convergence loss/accuracy, matched trajectories, performance, and the non-standard Actor execution timeline.

To regenerate existing plots only:

```bash
python -m part1_2.workflow artifacts
```

This also refreshes generated notes; inspect machine metadata in regenerated documentation before publication. Client `.npz` arrays, checkpoints, logs, and raw exception traces need separate publication review.

## Platform-specific helpers

`run_server.sh` has a fixed DeepDish5 working directory, Python environment/PYTHONPATH, GPU selection, and `flock` dependency. It is historical automation, not a portable quick start. `finish_validation.py` and `stop_duplicate.py` are operational recovery helpers, not standard reproduction entry points; do not invoke them as correctness tests. Use the documented module workflow with a compatible environment on another host. Part 2 and the combined report are available through the project overview.
