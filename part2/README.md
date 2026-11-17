# Part 2: Laplace Input Perturbation

[Project overview](../README.md) | [Evidence index](../part2_outputs/INDEX.md) | [Report](../report/README.md)

## Purpose and implementation

This completed component reuses sequential FedAvg from Part 1.1, without Ray. `noise.py` creates independent Laplace(0,b) pixel draws using NumPy PCG64 keys derived from seed, client ID, and scale. `runner.py` manages fixed or resampled perturbations, clean metrics, checkpoints, and recovery. `workflow.py` defines the grid and frozen test protocol; `validation.py` implements mathematical/scientific controls; `artifacts.py` and `documentation.py` generate plots/notes; `audit.py` checks completeness and preservation.

For raw image X, training uses X+noise, implemented as `(1-X)-noise` after the shared inversion. Main runs reuse one perturbed copy of each client's training data for all rounds, without clipping. Labels, validation, and test inputs remain clean. All reported training/validation/test accuracies evaluate clean inputs. At b=0, the code clones inputs without RNG draws. CPU-generated noise is transferred unchanged to CUDA.

## Dependencies, inputs, and prerequisites

Use Python, PyTorch, NumPy, Matplotlib, and psutil with the [recorded versions](../README.md#requirements). Ray is not imported here. Original GPU runs used DeepDish5's RTX 5060 Ti and two CPU threads; `--device cpu` is available, but exact GPU replay is environment-specific.

Place the instructor files under `Assignment3-data/`. Part 2 **requires** the existing `part1_1_deepdish5_outputs/splits.json`; it does not create a replacement manifest. Full scientific validation also needs the completed Part 1.1 C=0.10/E=3 runs at seeds 2026-2028. Complete any Part 1.2 work before Part 2 `init`, because initialization snapshots Part 1.1/1.2 code, outputs, and the input directory for preservation checking.

The shared data contain 100 clients and 62 classes, with 25,418 training, 6,407 validation, and 3,621 test examples. The model is 784-128-ReLU-62; all main runs use ten clients per round, E=3, learning rate 0.3, batch size 32, and 80 rounds.

## Full reproduction

Run these commands from the repository root in a separate reproduction copy. Keep historical data/results/checkpoints intact. For a fresh run, use a fresh `part2_outputs/` tree with completed prerequisite outputs; do not copy the historical preservation snapshot or frozen protocol into that new tree.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
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
python -m part2.workflow audit
```

- `init`: records the environment, protocol, split hash, and preservation snapshot.
- `math`: checks the noise distribution, dependence, keys, isolation, and a short zero-noise training control. It loads development data and is not purely an arithmetic-only command.
- `pilot`: three five-round runs at b=0, 0.1, 0.8.
- `grid`: all seven scales **0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80**, each at seeds **2026, 2027, 2028**.
- `science`: full zero-noise comparisons, repeated/noise-reuse runs, the synthetic signal control, extreme-noise and recovery tests, and client diagnostics. It depends on both the grid and pilots.
- `freeze`: fixes all 21 round-80 checkpoint hashes, the utility recommendation, and the separate Question 1 checkpoint before test scoring.
- `test`: scores those frozen checkpoints on held-out examples and caches progress/results.
- `artifacts`: regenerates plots and notes; `audit` checks saved evidence and writes audit/index files.

Replace GPU device arguments with `--device cpu` where needed. There is no output-directory option: paths are rooted at `part2_outputs/` relative to the working directory. Configurable experiments have a separate namespace and do not enter the main grid:

```bash
python -m part2.workflow train --device cpu --b 0.1 --seed 2026 --rounds 80 --epochs 3 --lr 0.3 --batch-size 32 --convention fixed --name custom_b01
python -m part2.workflow --help
```

Completed runs are cached by configuration; partial runs resume from the last committed checkpoint and regenerate identical fixed noise from client keys. Historical preservation snapshots include README/INDEX files. Subsequent documentation edits can invalidate those byte-for-byte checks without changing numerical results. Do not rewrite old snapshots to conceal differences; a fresh reproduction snapshots its own completed prerequisite tree.

## Scientific validation and interpretation

[validation/results.json](../part2_outputs/validation/results.json) records setup, expected behavior, observations, and limitations:

- One million samples per scale at b=0.025, 0.10, 0.80 test moments, seven quantiles, and tails against Laplace theory.
- Pixel correlations across 20,000 noise images have maximum tested absolute correlation **0.0206**.
- The zero-noise control exactly recovers all three Part 1.1 trajectories, selections, and final parameters.
- Key repeatability, unchanged clean inputs/labels, tiny/empty/missing-class clients, invalid scales, overflow, and interrupted-round recovery test mechanism boundaries.
- Fixed versus fresh-per-epoch ablations and extreme-noise failures test learning/numerical behavior. The synthetic signal reaches a 100% ceiling at every tested scale and cannot demonstrate general robustness.

The practical choice is **b=0.05**: mean final validation accuracy is **71.80%** versus **72.40%** at zero noise, a **0.60-percentage-point** decrease within the predefined 2pp utility tolerance. Mean round-80 test accuracy is **72.66%** over three seeds. The separate Question 1 b=0.10/seed-2026 validation-selected model scores **71.64%**; it is distinct from that scale's three-seed mean **70.97%**.

The utility tolerance is neither a significance criterion nor a privacy standard. Input distortion alone establishes no verified end-to-end epsilon-DP guarantee: labels/metadata are unchanged, reproducibility keys are public, clean metrics access unperturbed data, and multiple models/outputs require accounting. The report's hypothetical noisy-image calibrations are not guarantees for these trained models.

## Saved outputs and plots

[final_results.csv](../part2_outputs/final_results.csv) contains all 21 fixed-final-round scores; [seed_summary.csv](../part2_outputs/seed_summary.csv) contains means/sample SDs. `experiments/` holds histories, selections, noise diagnostics, per-client metrics, and initial/best/last checkpoints. The [index](../part2_outputs/INDEX.md) links the frozen protocol, results, scientific evidence, and eight PNG/PDF figure pairs.

```bash
python -m part2.workflow artifacts
```

This regenerates existing plots without training or held-out scoring. The image panel reads the development images and saved split; the other plots use saved numerical outputs. The non-standard heatmap shows each client's mean validation change relative to its matched zero-noise control, ordered by training-label entropy. Regeneration also rewrites notes; review generated Markdown and image-example redistribution rights before publication.

## Historical launchers

`run_server.sh` and `finalize_server.sh` retain fixed DeepDish5 working-directory and archive destinations, `flock`, and GPU checks. `PART2_PYTHON` and `PART2_PHYSICAL_GPU` override the interpreter/device but do not make the whole script portable. Use the module commands elsewhere. Keep raw course files, model checkpoints, transfer archives, and unreviewed host logs outside the public source package.
