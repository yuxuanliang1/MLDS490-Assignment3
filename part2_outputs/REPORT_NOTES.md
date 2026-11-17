# Assignment 3 Part 2: fixed Laplace input perturbations

## Scope and reference review

This Part 2 implementation uses the sequential FedAvg from Part 1.1, without Ray. The completed experiments preserve Part 1.1/1.2 numerical results, checkpoints, figures, and the supplied data. The combined [Overleaf report](../report/README.md) integrates all three parts. Training, scientific controls, and held-out evaluation were performed on DeepDish5.

The assignment Part 2 (page 4) specifies X prime = X + Laplace(0,b), C=.1, the two-layer 128-hidden-unit MLP, accuracy curves and train/test accuracy versus scale. General instructions (pages 1-2) emphasize edge/scientific validation (75% grading), a non-standard plot and a three-month extension plan. The Week 7 lecture supplies FedAvg/heterogeneity context; the Week 7 lab defines sensitivity and noise calibration. The provided Week 8 slides emphasize vertical/hybrid FL and homomorphic encryption, including privacy concerns about gradients; they are not a separate pixel-Laplace recipe. Their gradient-leakage discussion reinforces the need to distinguish input perturbation from protection of all model releases.

## Mechanism and implementation

For each original normalized image, $X\in[0,1]^{28\times28}$, sample independent pixel noise $\epsilon_{ij}\sim\operatorname{Laplace}(0,b)$, with $p(z)=\exp(-|z|/b)/(2b)$ for $b>0$, and use $X^\prime=X+\epsilon$. The theoretical mean is zero, variance $2b^2$, mean absolute noise $b$, and RMS $\sqrt{2}b$. At b=0 a new unchanged copy is returned without invoking an RNG.

Part 1.1 uses network inputs 1-X. Thus the exact raw-image perturbation is implemented as (1-X)-epsilon. There is no clipping, rotation, resizing or augmentation. Labels are unchanged int64; model computations use float32. Noise scales are in the supplied normalized pixel units. Float32 values outside [0,1] are accepted; negative/nonfinite scales, nonfinite inputs and overflow are rejected explicitly.

NumPy PCG64 creates each noise tensor on CPU using a dedicated SHA-256-derived key from experiment seed, client ID, exact float.hex(b) and a noise-domain tag. Samples are float32 and transferred unchanged to CPU/CUDA. Separate b and client keys generate independent streams; changing b never changes initialization, sampling or mini-batch order. A fixed noisy training tensor is created once per client/run and reused across all rounds. b=0 creates an unchanged copy. Noise hashes and MSE/range diagnostics are stored per client.

The required model remains Flatten -> Linear(784,128) -> ReLU -> Linear(128,62), returning raw logits. Local updates reuse the original Client.train: CrossEntropyLoss, shuffled mini-batch SGD, fresh model/optimizer per selected client, momentum zero. All ten selected clients begin from the identical global state. The server computes the original training-count-weighted parameter mean. No raw or noisy images are returned through aggregation.

## Data, evaluation and RNG controls

The original 100-user NumPy dataset and exact saved split manifest are reused: 25,418 training / 6,407 validation examples. No partitions are regenerated. For every round, including round zero, the updated model is evaluated on ALL clients' ORIGINAL CLEAN training and validation examples. Global loss and accuracy are sample-weighted sums, never unweighted averages of client accuracies. These clean training scores are comparable across scales; the noisy training objective is a different quantity.

Client selection uses the same independent keyed stream as Part 1.1 (seed, sampling, .1), with exactly ten distinct eligible clients per round. Local shuffle keys use seed, local, round and client ID. All b within a seed use identical initial weights, sampled clients and batch order. No specialized federated or DP framework is used. Sequential local updates and parameter aggregation are imported from Part 1.1 unchanged.

## Hardware and execution

Host `mlds-deepdish5`, Python 3.12.6, PyTorch 2.14.0+cu130, CUDA build 13.0, NumPy 2.5.3. GPU **NVIDIA GeForce RTX 5060 Ti**, compute capability [12, 0], memory 15.57 GiB. Physical GPU 0 is selected through its device UUID and exposed as cuda:0 with PCI ordering. nvidia-smi was checked before launch; the occupied RTX PRO GPU was excluded. The server has 48 logical CPUs and 93.8 GiB RAM; two PyTorch CPU threads are used.

Deterministic PyTorch algorithms, disabled TF32 and CUBLAS_WORKSPACE_CONFIG=:4096:8 match the earlier experiment. No new Python dependencies are needed. A single workflow lock prevents duplicate launches. Checkpoints and metric files are atomic. Interrupted training reconstructs fixed noise from keys, restores the committed model/sampling RNG/history, and replays an uncommitted round; failed/nonfinite updates never silently contribute a partial aggregate.

The 21 main runs total 290.04 seconds of training-loop/evaluation/checkpoint time, with maximum allocated PyTorch GPU memory 243.27 MiB. These timers exclude noise construction and initial evaluation, separately recorded as preparation_seconds; they also exclude SSH transfers and interpreter startup. Shared-server contention and filesystem I/O affect runtimes.

## Prespecified comparison and test protocol

Seven scales were predefined: 0, .025, .05, .1, .2, .4, .8. ALL seven were run for ALL three seeds 2026, 2027, 2028. Settings are fixed at C=.1 (10 clients), E=3, LR=.3, batch32, plain SGD, 80 rounds. Five-round pilots at b=0,.1,.8 check stability. No main noise level is discarded or silently clipped; no additional scale or hyperparameter is chosen from test performance. Stress scales and the fresh-noise ablation are separately labeled and excluded from the main grid.

predefined_protocol.json records the design before pilots/main training. After all development validation, frozen_protocol.json records all 21 FINAL ROUND-80 checkpoint hashes and the utility recommendation. Every final-round checkpoint is scored on the clean held-out aggregate test set exactly once; test progress is cached for safe restart. The required train/test-vs-b figure pairs the same round-80 model for both scores. All 21 test outcomes are retained; no favorable seed is selected. A separately predefined detailed Question1 result evaluates the validation-best checkpoint for b=.1, seed2026 and is labeled separately. Its test score is never mixed into the final-round plot.

The file was held out during this Part 2 development process. Parts 1.1/1.2 previously used the same supplied test source; Part 2 therefore follows a prespecified assessment protocol rather than claiming the source is newly unseen across the entire assignment. Repeated releases/analyses are a limitation; no Part 2 tuning follows test scoring.

## Final-round primary results: seed 2026

| b | Clean training | Clean validation | Best validation | Best round | Round-80 test | Correct / n | Loop seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 75.32% | 73.00% | 73.00% | 80 | 74.12% | 2684/3621 | 14.32 |
| 0.025 | 75.23% | 72.81% | 72.81% | 80 | 73.90% | 2676/3621 | 13.52 |
| 0.05 | 74.64% | 72.64% | 72.64% | 80 | 73.29% | 2654/3621 | 13.37 |
| 0.1 | 72.37% | 70.72% | 70.72% | 80 | 71.64% | 2594/3621 | 13.64 |
| 0.2 | 67.28% | 65.49% | 65.65% | 77 | 67.05% | 2428/3621 | 13.29 |
| 0.4 | 59.38% | 58.37% | 58.47% | 78 | 58.52% | 2119/3621 | 13.33 |
| 0.8 | 34.86% | 33.57% | 41.27% | 15 | 34.99% | 1267/3621 | 15.68 |

## Across-seed utility and variability

| b | Mean final clean training | Mean final validation +/- SD | Mean final test +/- SD | Validation drop vs b0 (pp) |
|---:|---:|---:|---:|---:|
| 0 | 74.97% | 72.40% +/- 0.57 pp | 73.38% +/- 0.66 pp | 0.00 |
| 0.025 | 74.83% | 72.13% +/- 0.60 pp | 73.16% +/- 0.65 pp | 0.27 |
| 0.05 | 74.15% | 71.80% +/- 0.73 pp | 72.66% +/- 0.55 pp | 0.60 |
| 0.1 | 72.16% | 70.05% +/- 0.58 pp | 70.97% +/- 0.57 pp | 2.35 |
| 0.2 | 66.55% | 64.76% +/- 0.83 pp | 66.06% +/- 0.90 pp | 7.64 |
| 0.4 | 58.91% | 57.98% +/- 0.50 pp | 58.79% +/- 0.54 pp | 14.42 |
| 0.8 | 36.77% | 35.57% +/- 1.76 pp | 37.15% +/- 1.92 pp | 36.83 |

The largest across-seed validation SD is 1.76 pp at b=0.8, compared with 0.57 pp at b=0. Variability is not monotone at every intermediate scale; three seeds do not establish a universal variance law.

SD is the sample standard deviation over three seeds, not a confidence interval. Seed changes model initialization, client participation, batch order and perturbation; within each seed these non-noise components are matched across scales. Three seeds provide limited uncertainty evidence; paired comparisons are more informative than treating the runs as unrelated.

## Convergence and representative positive-noise experiment

- b=0: clean train CE 4.126150 -> 0.865133; validation CE 4.126700 -> 1.012831; clean train accuracy 3.19% -> 75.32%; validation accuracy 3.03% -> 73.00%. Largest one-round validation drop: 1.389 pp; last-ten-round sample SD: 0.598 pp.
- b=0.1: clean train CE 4.126150 -> 0.986190; validation CE 4.126700 -> 1.119120; clean train accuracy 3.19% -> 72.37%; validation accuracy 3.03% -> 70.72%. Largest one-round validation drop: 0.983 pp; last-ten-round sample SD: 0.590 pp.
- b=0.8: clean train CE 4.126150 -> 3.325932; validation CE 4.126700 -> 3.344604; clean train accuracy 3.19% -> 34.86%; validation accuracy 3.03% -> 33.57%. Largest one-round validation drop: 3.668 pp; last-ten-round sample SD: 1.185 pp.

For Question1, the separately labeled best-validation b=.1 seed2026 checkpoint is round 80, with held-out test **71.64% (2594/3621)**. The main comparison uses its round80 result instead. Final-versus-best gaps and per-round fluctuations are preserved in CSV/JSON; fixed 80-round training does not prove optimization convergence. Trajectory SD combines learning trends and fluctuations, unlike the across-seed SD.

## Perturbation magnitude and utility interpretation

| b | Measured RMS pixel distortion | Outside [0,1] fraction | Ink-energy SNR (dB) |
|---:|---:|---:|---:|
| 0 | 0.000000 | 0.00% | infinite (no noise) |
| 0.025 | 0.035364 | 46.61% | 13.265075345993484 |
| 0.05 | 0.070722 | 47.10% | 7.245220234389792 |
| 0.1 | 0.141431 | 47.63% | 1.2253703388662696 |
| 0.2 | 0.282891 | 48.62% | -4.796120335983077 |
| 0.4 | 0.565786 | 53.15% | -10.816765558067669 |
| 0.8 | 1.131269 | 63.92% | -16.83504237782034 |

**Validation-based practical recommendation: b=0.05.** This is the largest predefined positive noise scale within 2 percentage points of the clean baseline's mean final validation accuracy. Its mean validation score is 71.80%, a reduction of 0.60 pp. The tolerance is an explicit engineering choice, not a privacy guarantee. The recommendation was written before Part 2 test scoring. If a stricter utility tolerance or a formally specified privacy requirement applies, the choice must be reconsidered on development data with a new protocol.

Greater b increases expected absolute distortion linearly and variance quadratically. Compare the actual accuracy/SD rows above: finite-sample results need not be monotone at every scale or seed. Fixed noisy examples can be memorized, so training on them is not equivalent to fresh-noise augmentation. The image panel uses the same display range for all columns; saturated white/black values in the plot are display limits only, not clipped training data. Ink-energy SNR is a visual/utility diagnostic, not a measure of privacy leakage.

## Scientific validation A-J and edge-case evidence

### Controlled-signal interpretation

The synthetic spatial signal reaches 100% clean validation accuracy in all nine runs (b=0,.2,2; three seeds). This particular control does NOT demonstrate degradation at its tested scales: it has a ceiling effect. The spatial feature is repeated over several pixels and there are 480 noisy training examples, so the global learner can retain the class direction even at substantial noise. Validation inputs are clean. This limits the control's ability to detect a robustness boundary and does not establish general robustness; the 62-class real-data comparison and extreme-noise stress tests show substantial utility loss. A future control should vary signal strength and measure a less saturated clean-validation task, with its protocol specified independently of test outcomes.

For the separate 40-round b=0.1 ablation, fixed-noise validation is 64.80%, versus 65.83% for fresh-per-local-epoch noise: +1.03 pp. This is a measured training-procedure difference, not a reason to mix the ablation into the fixed-noise grid.
For the separate 40-round b=0.4 ablation, fixed-noise validation is 55.08%, versus 60.15% for fresh-per-local-epoch noise: +5.07 pp. This is a measured training-procedure difference, not a reason to mix the ablation into the fixed-noise grid.

The largest finite local stress case b=100 has validation CE 9.029e+22, accuracy 0.00%, and mean maximum softmax confidence 0.8774. Finite parameters and high confidence therefore do not imply useful learning. Explicitly rejected stress scales: [1000.0, 1000000.0]; failure messages remain in the evidence below.

At b=0.1, correlations of paired client accuracy change with training size, entropy and class diversity are -0.038, -0.230 and -0.279. The smallest-size quartile has mean change -2.57 pp versus -2.52 pp in the largest-size quartile. Smaller clients are not uniformly more sensitive in these observations. Baseline accuracy, class composition, writer difficulty and validation sampling noise can confound these associations.

At b=0.8, correlations of paired client accuracy change with training size, entropy and class diversity are -0.174, 0.093 and -0.148. The smallest-size quartile has mean change -34.60 pp versus -39.22 pp in the largest-size quartile. Smaller clients are not uniformly more sensitive in these observations. Baseline accuracy, class composition, writer difficulty and validation sampling noise can confound these associations.

### A_Laplace_distribution

**Hypothesis:** IID Laplace draws have the specified moments, tails and quantiles

**Setup:** 1,000,000 draws per scale .025,.1,.8; float32 samples, float64 statistics

**Expected / criterion:** All moment, quantile and tail deviations within 6 approximate sampling SE

**Observed status:** PASS / investigation completed. Mandatory correctness gate.

```json
[
  {
    "b": 0.025,
    "n": 1000000,
    "mean": -1.2865619232815125e-05,
    "variance": 0.0012485096783031485,
    "theoretical_variance": 0.0012500000000000002,
    "mean_absolute": 0.024960219558961768,
    "quantile_probabilities": [
      0.01,
      0.1,
      0.25,
      0.5,
      0.75,
      0.9,
      0.99
    ],
    "empirical_quantiles": [
      -0.09795591190457344,
      -0.0402015957981348,
      -0.01728019444271922,
      -6.109117521191365e-06,
      0.017241950146853924,
      0.04020179733633995,
      0.09769735485315323
    ],
    "theoretical_quantiles": [
      -0.09780057513570366,
      -0.04023594781085251,
      -0.01732867951399863,
      -0.0,
      0.01732867951399863,
      0.04023594781085252,
      0.09780057513570363
    ],
    "tail_probability": 0.04971,
    "standard_errors": {
      "mean_z": 0.3638946641475056,
      "variance_z": 0.5331936996005203,
      "absolute_mean_z": 1.5912176415293355,
      "max_quantile_z": 2.0029289387057143,
      "tail_z": 0.35432958558478206
    }
  },
  {
    "b": 0.1,
    "n": 1000000,
    "mean": 1.731379619797663e-05,
    "variance": 0.019973362190983134,
    "theoretical_variance": 0.020000000000000004,
    "mean_absolute": 0.10002555193254649,
    "quantile_probabilities": [
      0.01,
      0.1,
      0.25,
      0.5,
      0.75,
      0.9,
      0.99
    ],
    "empirical_quantiles": [
      -0.3901090794801712,
      -0.1612955346703529,
      -0.06924429535865784,
      5.992302430968266e-05,
      0.06928596831858158,
      0.16118527203798294,
      0.3896318602561951
    ],
    "theoretical_quantiles": [
      -0.39120230054281463,
      -0.16094379124341004,
      -0.06931471805599453,
      -0.0,
      0.06931471805599453,
      0.16094379124341007,
      0.3912023005428145
    ],
    "tail_probability": 0.049737,
    "standard_errors": {
      "mean_z": 0.12242702699671137,
      "variance_z": 0.5956395173337894,
      "absolute_mean_z": 0.25551932546483136,
      "max_quantile_z": 1.5783518746590146,
      "tail_z": 0.23019436544257987
    }
  },
  {
    "b": 0.8,
    "n": 1000000,
    "mean": -0.00024117511784259277,
    "variance": 1.2836539839934673,
    "theoretical_variance": 1.2800000000000002,
    "mean_absolute": 0.8009903429202121,
    "quantile_probabilities": [
      0.01,
      0.1,
      0.25,
      0.5,
      0.75,
      0.9,
      0.99
    ],
    "empirical_quantiles": [
      -3.1325106596946717,
      -1.2909656643867493,
      -0.5548799484968185,
      0.0004159624659223482,
      0.5548042356967926,
      1.2883132457733153,
      3.1365076375007637
    ],
    "theoretical_quantiles": [
      -3.129618404342517,
      -1.2875503299472804,
      -0.5545177444479562,
      -0.0,
      0.5545177444479562,
      1.2875503299472806,
      3.129618404342516
    ],
    "tail_probability": 0.04978,
    "standard_errors": {
      "mean_z": 0.21317070159995252,
      "variance_z": 1.2766494684513263,
      "absolute_mean_z": 1.2379286502650633,
      "max_quantile_z": 1.4230560164453971,
      "tail_z": 0.0324975333643288
    }
  }
]
```

**Interpretation:** Scale-invariant normalized moments agree with Laplace theory

**Limitations:** Normal/delta-method SE are approximate; multiple tests use conservative 6-SE bounds

### B_pixel_independence

**Hypothesis:** Distinct pixel streams have near-zero correlations

**Setup:** 20,000 independently perturbed images; 496 pairs across 32 spread pixel positions

**Expected / criterion:** Maximum |r| < 6/sqrt(20000), matching shape 20000x28x28

**Observed status:** PASS / investigation completed. Mandatory correctness gate.

```json
{
  "max_abs_correlation": 0.020592284554336725,
  "mean_correlation": -0.0001680184248325483,
  "threshold": 0.04242640687119285,
  "shape": [
    20000,
    28,
    28
  ]
}
```

**Interpretation:** No material linear dependence detected across tested positions

**Limitations:** Correlation is not a proof of independence; generator algorithm supplies independence

### D_data_isolation

**Hypothesis:** Perturbation changes training pixels only

**Setup:** Hash every client training input/label and clean validation tensor; hash held-out FILE bytes without loading examples

**Expected / criterion:** All 100 clients retain original tensors, labels, validation and boundaries; test file hash unchanged

**Observed status:** PASS / investigation completed. Mandatory correctness gate.

```json
{
  "clients_checked": 100,
  "all_checks": true,
  "test_file_sha256": "0df3082c14f7b62f6f5a264181fd7a243533f9463562f5986addf2f80b4060b2"
}
```

**Interpretation:** Original client-specific tensors are immutable inputs; aggregation imports unchanged parameter-only interface

**Limitations:** Shared-machine logical boundary is not a security sandbox

### C_zero_noise_equivalence

**Hypothesis:** b=0 preserves Part1.1 FedAvg exactly

**Setup:** Three real rounds; same architecture, weights, sampling, shuffle seeds, E3 LR.3 batch32, same GPU

**Expected / criterion:** Every recorded metric and final parameter equal bitwise

**Observed status:** PASS / investigation completed. Mandatory correctness gate.

```json
{
  "max_parameter_difference": 0.0,
  "metric_differences": {
    "train_loss": 0.0,
    "validation_loss": 0.0,
    "train_accuracy": 0.0,
    "validation_accuracy": 0.0
  },
  "selections_equal": true
}
```

**Interpretation:** New evaluation/noise plumbing does not change clean FedAvg

**Limitations:** Short check; all 80-round zero-noise runs are also compared with saved Part1.1 checkpoints later

### I_noise_reproducibility_and_edges

**Hypothesis:** Keyed CPU-generated noise transfers identically across devices

**Setup:** Repeat keys, vary experiment seed/client ID, copy b0, reject negative/nonfinite scales and overflowing float32 samples

**Expected / criterion:** Identical same keys and CPU/GPU; different seeds/clients differ; invalid inputs explicitly fail

**Observed status:** PASS / investigation completed. Mandatory correctness gate.

```json
{
  "cpu_gpu_max_difference": 0.0,
  "repeat_equal": true,
  "different_seed": true,
  "different_client": true,
  "zero_copy_equal": true,
  "zero_copy_distinct": true,
  "invalid_rejected": [
    "-1.0",
    "nan",
    "inf"
  ],
  "overflow_rejected": true,
  "nonfinite_input_rejected": true
}
```

**Interpretation:** Noise RNG is independent of model, sampling and shuffle RNGs; b=0 takes a clone-only path

**Limitations:** CPU generation is deliberate; does not claim identical native CUDA PRNG algorithms

### C_full_zero_control

**Hypothesis:** Full no-noise trajectories reproduce saved Part1.1 results

**Setup:** Compare all 81 metric rows, 800 selections and round80 parameters for three seeds

**Expected / criterion:** Bitwise zero differences

**Observed status:** PASS / investigation completed. Mandatory correctness gate.

```json
[
  {
    "seed": 2026,
    "max_parameter_difference": 0.0,
    "max_metric_difference": 0.0,
    "selections_equal": true
  },
  {
    "seed": 2027,
    "max_parameter_difference": 0.0,
    "max_metric_difference": 0.0,
    "selections_equal": true
  },
  {
    "seed": 2028,
    "max_parameter_difference": 0.0,
    "max_metric_difference": 0.0,
    "selections_equal": true
  }
]
```

**Interpretation:** Validates full-run reuse across noise plumbing

**Limitations:** Identical software/GPU architecture required for bitwise agreement

### E_controlled_noise_degradation

**Hypothesis:** Increasing training noise weakens learning a small spatial signal

**Setup:** 20 synthetic clients, 480 train/160 validation, three seeds, b0/.2/2, 30 rounds E1 LR.1

**Expected / criterion:** Compare seed-averaged validation, without requiring each run to decrease monotonically

**Observed status:** PASS / investigation completed. Descriptive investigations do not impose an unwarranted universal accuracy ordering.

```json
[
  {
    "seed": 371,
    "b": 0.0,
    "validation": 1.0,
    "finite": true
  },
  {
    "seed": 371,
    "b": 0.2,
    "validation": 1.0,
    "finite": true
  },
  {
    "seed": 371,
    "b": 2.0,
    "validation": 1.0,
    "finite": true
  },
  {
    "seed": 372,
    "b": 0.0,
    "validation": 1.0,
    "finite": true
  },
  {
    "seed": 372,
    "b": 0.2,
    "validation": 1.0,
    "finite": true
  },
  {
    "seed": 372,
    "b": 2.0,
    "validation": 1.0,
    "finite": true
  },
  {
    "seed": 373,
    "b": 0.0,
    "validation": 1.0,
    "finite": true
  },
  {
    "seed": 373,
    "b": 0.2,
    "validation": 1.0,
    "finite": true
  },
  {
    "seed": 373,
    "b": 2.0,
    "validation": 1.0,
    "finite": true
  }
]
```

**Interpretation:** Measured controlled responses are reported rather than assuming monotonic accuracy

**Limitations:** Synthetic signal and small sample size differ from real handwriting

### I_training_reproducibility

**Hypothesis:** Same seed reproduces positive-noise training

**Setup:** Two independent fixed b=.1 datasets and five-round runs

**Expected / criterion:** Exactly equal parameters and recorded metrics

**Observed status:** PASS / investigation completed. Mandatory correctness gate.

```json
{
  "max_parameter_difference": 0.0,
  "history_metrics_equal": true
}
```

**Interpretation:** Fixed noise and keyed local/sampling streams permit deterministic replay

**Limitations:** Only one positive noise scale is independently repeated

### J_fixed_vs_resampled

**Hypothesis:** Fresh per-epoch noise changes the optimization distribution

**Setup:** Matched real data, same 40 rounds E3 LR.3 batch32 and sampling for b=.1,.4

**Expected / criterion:** Quantify difference; no universal ordering assumed

**Observed status:** PASS / investigation completed. Descriptive investigations do not impose an unwarranted universal accuracy ordering.

```json
[
  {
    "b": 0.1,
    "convention": "fixed",
    "final_validation": 0.6480412049321055,
    "best_validation": 0.6530357421570158
  },
  {
    "b": 0.1,
    "convention": "resampled",
    "final_validation": 0.6583424379584829,
    "best_validation": 0.6614640237240518
  },
  {
    "b": 0.4,
    "convention": "fixed",
    "final_validation": 0.550803808334634,
    "best_validation": 0.550803808334634
  },
  {
    "b": 0.4,
    "convention": "resampled",
    "final_validation": 0.6015295770251288,
    "best_validation": 0.6077727485562666
  }
]
```

**Interpretation:** This is a separate stochastic-training ablation and excluded from the primary fixed-noise/test grid

**Limitations:** One seed and 40 rounds; no formal privacy accounting

### H_high_noise_stability

**Hypothesis:** Large perturbations may degrade numerical stability and useful confidence

**Setup:** One real small client, 3 local epochs, scales up to1e6 (stress only); parameter hooks inspect gradients; no clipping

**Expected / criterion:** Record finite deterioration or explicit rejection, never replace a failed update

**Observed status:** PASS / investigation completed. Descriptive investigations do not impose an unwarranted universal accuracy ordering.

```json
[
  {
    "b": 1e-12,
    "status": "finite",
    "max_abs_pixel": 1.0,
    "max_abs_gradient": 0.1417524516582489,
    "validation_loss": 3.800309499104818,
    "validation_accuracy": 0.041666666666666664,
    "mean_prediction_confidence": 0.1256873905658722
  },
  {
    "b": 1.6,
    "status": "finite",
    "max_abs_pixel": 17.887428283691406,
    "max_abs_gradient": 0.2442828118801117,
    "validation_loss": 3.939697583516439,
    "validation_accuracy": 0.08333333333333333,
    "mean_prediction_confidence": 0.027346795424818993
  },
  {
    "b": 3.2,
    "status": "finite",
    "max_abs_pixel": 40.71889114379883,
    "max_abs_gradient": 3.770695686340332,
    "validation_loss": 4.162240346272786,
    "validation_accuracy": 0.0,
    "mean_prediction_confidence": 0.018869390711188316
  },
  {
    "b": 10.0,
    "status": "finite",
    "max_abs_pixel": 127.80247497558594,
    "max_abs_gradient": 10713.5263671875,
    "validation_loss": 1193857.1666666667,
    "validation_accuracy": 0.0,
    "mean_prediction_confidence": 1.0
  },
  {
    "b": 100.0,
    "status": "finite",
    "max_abs_pixel": 1104.97412109375,
    "max_abs_gradient": 11215191408640.0,
    "validation_loss": 9.028997277417415e+22,
    "validation_accuracy": 0.0,
    "mean_prediction_confidence": 0.8773738741874695
  },
  {
    "b": 1000.0,
    "status": "explicit_failure",
    "error": "Non-finite local loss, client 25",
    "max_abs_pixel": 12538.2060546875
  },
  {
    "b": 1000000.0,
    "status": "explicit_failure",
    "error": "Non-finite local loss, client 25",
    "max_abs_pixel": 10642062.0
  }
]
```

**Interpretation:** High-noise failures are retained; finite softmax confidence alone does not establish predictive quality

**Limitations:** Local stress test is not a full large-b FedAvg trajectory

### edge_tiny_and_resume

**Hypothesis:** Missing classes, size imbalance and interrupted runs remain well-defined

**Setup:** Synthetic clients n=0,1,2,7,101,23,3,9,11,13, class0 only; one-round stop/resume

**Expected / criterion:** 9 eligible clients; resumed/uninterrupted parameters equal

**Observed status:** PASS / investigation completed. Mandatory correctness gate.

```json
{
  "eligible_clients": 9,
  "max_resumption_difference": 0.0
}
```

**Interpretation:** Empty clients are ineligible; tiny nonempty clients keep their original samples and missing classes

**Limitations:** These are synthetic stress fixtures, not observed empty real clients

### G_client_noise_sensitivity

**Hypothesis:** Federated clients may lose unequal utility under perturbation

**Setup:** 100 original clients; per-seed final validation differences relative to matching b0, averaged over three seeds

**Expected / criterion:** Quantify changes and associations; no causal/pass threshold for correlations

**Observed status:** PASS / investigation completed. Descriptive investigations do not impose an unwarranted universal accuracy ordering.

```json
{
  "correlations": [
    {
      "b": 0.1,
      "pearson_correlations": {
        "train_n": -0.03846247939931159,
        "entropy_bits": -0.23046900764447376,
        "observed_classes": -0.27921538325724027
      },
      "quartile_changes": {
        "train_n": {
          "bottom_quartile_change": -0.025658805391643173,
          "top_quartile_change": -0.02516720943791186
        },
        "entropy_bits": {
          "bottom_quartile_change": -0.016278481075908305,
          "top_quartile_change": -0.02710487622441998
        }
      }
    },
    {
      "b": 0.8,
      "pearson_correlations": {
        "train_n": -0.17430943450534153,
        "entropy_bits": 0.09252019731956662,
        "observed_classes": -0.14768090355511415
      },
      "quartile_changes": {
        "train_n": {
          "bottom_quartile_change": -0.3459951482472458,
          "top_quartile_change": -0.39224432531080483
        },
        "entropy_bits": {
          "bottom_quartile_change": -0.37121126690822115,
          "top_quartile_change": -0.36352035000906163
        }
      }
    }
  ],
  "largest_degradations": {
    "0.1": [
      {
        "client_id": 72,
        "mean_change": -0.08928571428571426,
        "train_n": 223,
        "entropy_bits": 5.020948880141732,
        "observed_classes": 51
      },
      {
        "client_id": 43,
        "mean_change": -0.06451612903225808,
        "train_n": 244,
        "entropy_bits": 5.233936831075433,
        "observed_classes": 55
      },
      {
        "client_id": 41,
        "mean_change": -0.06432748538011694,
        "train_n": 228,
        "entropy_bits": 5.126967971137422,
        "observed_classes": 51
      }
    ],
    "0.8": [
      {
        "client_id": 27,
        "mean_change": -0.6833333333333335,
        "train_n": 156,
        "entropy_bits": 4.682227515231736,
        "observed_classes": 40
      },
      {
        "client_id": 23,
        "mean_change": -0.5666666666666668,
        "train_n": 237,
        "entropy_bits": 5.09908513577359,
        "observed_classes": 48
      },
      {
        "client_id": 43,
        "mean_change": -0.5268817204301075,
        "train_n": 244,
        "entropy_bits": 5.233936831075433,
        "observed_classes": 55
      }
    ]
  }
}
```

**Interpretation:** The heatmap orders clients by fixed training-label entropy; size/diversity associations are observational

**Limitations:** Small local validation samples and writer/class differences confound associations; no causal or formal privacy interpretation

### F_image_distortion

**Hypothesis:** Independent pixel noise obscures ink at increasing scale

**Setup:** Actual per-pixel perturbation MSE and range violations from all 21 fixed datasets; display four fixed training images with shared [0,1] display scale

**Expected / criterion:** MSE near 2b^2; noise and out-of-range values remain unmodified in training

**Observed status:** PASS / investigation completed. Descriptive investigations do not impose an unwarranted universal accuracy ordering.

```json
[
  {
    "b": 0.0,
    "mean_squared_distortion": 0.0,
    "rms_distortion": 0.0,
    "mean_outside_unit_fraction": 0.0,
    "ink_signal_power": 0.026523285149481803,
    "ink_snr_db": null
  },
  {
    "b": 0.025,
    "mean_squared_distortion": 0.0012506039031248213,
    "rms_distortion": 0.0353638785079468,
    "mean_outside_unit_fraction": 0.46605340036739135,
    "ink_signal_power": 0.026523285149481803,
    "ink_snr_db": 13.265075345993484
  },
  {
    "b": 0.05,
    "mean_squared_distortion": 0.0050015577871703595,
    "rms_distortion": 0.07072169247953812,
    "mean_outside_unit_fraction": 0.4710301057711523,
    "ink_signal_power": 0.026523285149481803,
    "ink_snr_db": 7.245220234389792
  },
  {
    "b": 0.1,
    "mean_squared_distortion": 0.020002776411486983,
    "rms_distortion": 0.14143117199361316,
    "mean_outside_unit_fraction": 0.4762886794287943,
    "ink_signal_power": 0.026523285149481803,
    "ink_snr_db": 1.2253703388662696
  },
  {
    "b": 0.2,
    "mean_squared_distortion": 0.08002751804134925,
    "rms_distortion": 0.282891353776232,
    "mean_outside_unit_fraction": 0.4861884441784728,
    "ink_signal_power": 0.026523285149481803,
    "ink_snr_db": -4.796120335983077
  },
  {
    "b": 0.4,
    "mean_squared_distortion": 0.32011341180729663,
    "rms_distortion": 0.5657856588915069,
    "mean_outside_unit_fraction": 0.5314817350548008,
    "ink_signal_power": 0.026523285149481803,
    "ink_snr_db": -10.816765558067669
  },
  {
    "b": 0.8,
    "mean_squared_distortion": 1.2797689002966635,
    "rms_distortion": 1.1312687126835355,
    "mean_outside_unit_fraction": 0.6391742800166725,
    "ink_signal_power": 0.026523285149481803,
    "ink_snr_db": -16.83504237782034
  }
]
```

**Interpretation:** Ink-energy SNR uses signal 1-X to avoid white-background energy dominating the ratio; display saturation is not training clipping

**Limitations:** Images and distortion demonstrate information loss but do not measure a privacy parameter

## Edge-case policy and actual versus synthetic cases

- Actual supplied data: all 100 clients are eligible; training sizes vary and many classes are absent locally. Fixed noisy samples routinely fall outside [0,1]; no clipping is used. Noise-induced loss/accuracy changes are measured on clean full pools.
- b=0 returns a distinct unchanged copy without RNG use. Very small b (1e-12) is allowed; some additions round away in float32. Positive noise scales preserve global labels and local indices.
- Negative b, NaN/Inf b, nonfinite inputs and float32 overflow are explicitly rejected. Failed local losses/parameters are rejected by the reused Client/aggregate code; the high-noise stress check additionally inspects gradient finiteness. No bad update is substituted.
- n=0 clients are ineligible; tiny nonempty clients retain their samples without duplicated/fabricated examples. Missing classes retain the global 62-class output space. Tiny/empty/extreme-size fixtures are synthetic stress cases, not claims about observed empty real clients.
- CPU and GPU noise equality is achieved by generating on CPU then transferring the same float32 draws. Native CPU/CUDA random algorithms are not assumed identical. Fixed and resampled noise are distinct procedures and are not pooled in the main results.

## Privacy interpretation and limits

For a specified query with L1 sensitivity Delta1, the classical Laplace mechanism calibrates scale as b=Delta1/epsilon. This result requires explicit neighboring datasets, a released query/mechanism, and a valid sensitivity bound; see [Dwork and Roth, The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) and [NIST SP 800-226](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-226.pdf). Input perturbation here changes images before local SGD; it is not gradient perturbation, update perturbation, or DP-SGD.

No epsilon is attributed to the actual trained models or complete FedAvg pipeline. The report separately provides hypothetical noisy-image calibrations under explicitly defined adjacency and private randomness; these do not establish an implemented-system guarantee. In this experiment, labels, sample counts and client IDs are unchanged; image-only reasoning would not automatically protect them. Public experiment seeds make noise reconstructible. Clean training/validation metrics directly depend on unperturbed data, so their release is not merely postprocessing of a noisy dataset. Multiple model/checkpoint releases across scales/seeds would require a defined joint release analysis and appropriate composition. Fixed-noise reuse and fresh-noise-per-epoch training also require different release/accounting reasoning. A deployment would need a clear threat model, protected unit (image or writer), sensitivity/contribution bounds, suitable randomness, a proof/accountant for every released output, and empirical attack assessment. Our utility and distortion diagnostics alone establish none of those guarantees.

## Limitations and three-month extension plan

The architecture and fixed 80-round budget constrain attainable utility. The same original users and small local validation sets limit fairness conclusions. Entropy describes label mix but not writer style. Three seeds and one main LR do not characterize all optimization/noise interactions. High-b local stress tests are not full high-b training runs. No formal privacy attack, privacy accountant or record/user-level DP proof is implemented. The optional resampling ablation uses one seed/40 rounds.

1. **Month 1: utility and statistical reliability.** Prespecify 10-20 seeds, paired client-bootstrap uncertainty, more scales near the validation tolerance boundary, fixed-compute comparisons and rare-class/calibration analyses. Extend deterministic resume/failure tests across devices, and cross-check noise-generation moments/tails using an independent implementation.
2. **Month 2: heterogeneity and robustness.** Measure writer-level errors, class-conditional sensitivity and attack risk; run matched client-size/label-skew interventions, fixed/resampled noise with more seeds, controlled missing classes, safe high-noise/gradient stress tests and longer training. Separate local memorization from clean generalization.
3. **Month 3: formal privacy and confirmatory evaluation.** Specify adjacency/protected unit and the full release surface, bound contributions/sensitivity, choose appropriate privacy randomness and derive/account the actual mechanism including labels/metrics/model releases. Only then compare utility at meaningful formal privacy budgets. Pre-register conclusions and use a new untouched writer-level evaluation source instead of continuing to tune against the assignment test set.

## Figures and reproduction

- `dp_accuracy_convergence.png` / `.pdf`: Clean train/validation curves for b=0 and b=0.1; includes round zero.
- `dp_loss_convergence.png` / `.pdf`: Clean train/validation curves for b=0 and b=0.1; includes round zero.
- `dp_accuracy_vs_noise_scale.png` / `.pdf`: Both curves use each configuration's round-80 checkpoint; no best/final checkpoint mixing.
- `dp_noise_seed_stability.png` / `.pdf`: All three prespecified seeds are reported; sample SD is not a confidence interval.
- `dp_noise_distribution_validation.png` / `.pdf`: Moment, quantile and tail criteria use sampling uncertainty; histograms show the normalized central range.
- `dp_image_perturbations.png` / `.pdf`: First four saved training indices of client 0; positive raw-image noise is applied before 1-X transformation.
- `dp_client_robustness_heatmap.png` / `.pdf`: Non-standard client-level utility diagnostic; ordering uses training-label entropy, not test data.
- `dp_privacy_utility_tradeoff.png` / `.pdf`: Noise magnitude is not epsilon; recommendation uses validation utility only.

All eight plots are 300-DPI PNG/vector PDF pairs. The two noise-scale accuracy plots use explicitly labeled symlog horizontal spacing (linear below b=.025) to keep every small scale readable. See ../part2/README.md for exact commands, caching/resumption, and the indexed result archive.

Publication note: personal paths and host addresses in these Markdown excerpts are omitted. Original numerical JSON/CSV evidence is unchanged. Generated notes can restore machine metadata; consult the component README before regeneration or publication.
