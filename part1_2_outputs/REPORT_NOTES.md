# Assignment 3 Part 1.2: Ray Core parallel clients

## Design and preserved components

Part 1.1 source/results and supplied data/references are preserved. This implementation imports its MLP (784-128-ReLU-62), keyed local SGD, weighted aggregation, sample-weighted evaluation, and checkpoint helpers. The saved split manifest is reused without modification: 100 clients, 25,418 train and 6,407 validation examples. Images use 1-x. Every required round uniformly samples exactly four distinct clients, creates exactly four fresh Ray Core Actors, broadcasts identical CPU model parameters, runs E=3 plain SGD LR=.3 batch32 concurrently, gathers all updates, aggregates in sampled order, evaluates the full pool, and terminates every Actor before the next round. Round zero is included.

Each Actor reads only its assigned client NPZ file. It returns parameters/counts/metadata, never raw examples. The shared filesystem is a logical simulation boundary, not an adversarial access-control guarantee. The server uses client-side metric summaries from the preserved evaluator. A common 150ms release barrier follows Actor readiness; this is measured orchestration overhead, not artificial training duration.

CPU Actors reserve one CPU and zero GPU; fractional GPU Actors reserve one CPU and .25 GPU each. Ray starts a private four-CPU runtime and exposes zero GPUs for CPU mode or only the verified physical GPU UUID for GPU mode. No existing Ray cluster is joined or stopped. Finally blocks kill all Actors, wait for CPU resource return AND OS PID exit; incomplete rounds never aggregate or commit. Retries are explicit: restart replays the uncommitted round with identical keyed seeds.

## Environment and execution choice

Host mlds-deepdish5; Python 3.12.6; PyTorch 2.14.0+cu130; Ray 2.60.0; CUDA build 13.0. Physical GPU 0 was an idle RTX 5060 Ti (16 GB), verified by UUID; busy GPU 1 was excluded. See nvidia_before/after and per-Actor placement records. Chosen mode: **cpu**, by lower median complete Actor-round time among numerically valid strategies. CPU driver uses one thread; four CPU Actors each use one thread (four concurrent cores). GPU allocation is logical and does not isolate memory.

## Training, numerical equivalence and final test

80 rounds: training accuracy 73.5738%, validation 70.7039%; best validation 72.2959% at round 78. The five-round real pilot used the same configuration; no test-based adjustment occurred.

The full matched sequential run uses the same 80 selections, initial weights, split, E, LR, batches and seed. Asynchronous finish order cannot change this synchronous algorithm: aggregation waits for all four and preserves order. Floating-point summation is not reordered by completion time. Full trajectory evidence:
```json
{
  "final_max_parameter_difference": 0.0,
  "max_metric_difference": 0.0,
  "identical_selections": true
}
```

Frozen validation-selected checkpoint: parallel/best.pt, round 78. Held-out result: **73.7918% (2672/3621)**, CE 1.024556. Selection/checkpoint hashes were saved before this Part 1.2 test access. No settings were revised using this result. Part 1.1's 10-client score is not a target or a fair algorithm comparison against this four-client setting.

## Performance and interpretation

| Strategy | Mean round s | Sample SD s | Warm mean s | Ray startup s | 3-round end-to-end s | CPU-reference speedup |
|---|---:|---:|---:|---:|---:|---:|
| sequential | 0.03342 | 0.00173 | 0.03248 | 0.0000 | 0.1003 | 1.00000 |
| cpu | 3.00794 | 0.41718 | 3.24764 | 5.1108 | 14.1347 | 0.01111 |
| cuda:0 | 4.11489 | 0.69292 | 3.71499 | 3.3404 | 15.6850 | 0.00812 |

Full-workload speedup (sequential wall / Ray wall including runtime startup) = **0.04327x**. All measured values include fresh Actor lifecycle costs; the three-round end-to-end comparison adds Ray runtime initialization once, not once per round. Benchmark repetition 0 includes first-worker effects; warm columns use repetitions 1-2. The full workload additionally includes every global evaluation and checkpoint I/O. No speedup is inferred merely from concurrency.

Sequential CPU uses one thread; Ray CPU uses four one-thread processes on the same host. GPU mode uses a different accelerator resource and is a strategy benchmark, not an equal-hardware CPU speedup claim. Driver-side evaluation stays CPU for both full trajectories. Shared server load and filesystem performance are uncontrolled; only three timing repeats were used. Startup imports, optimizer construction, serialization, process termination, and the measured release barrier overwhelm this small model. GPU Actors additionally pay context initialization and share one device without memory isolation.

## Scientific validation A-G

### A_D_equivalence_cpu

Hypothesis: Parallelism preserves local SGD and weighted synchronous FedAvg. Design: Unequal real clients 25,67,16,36; identical CPU initialization and keyed batch order; E3 LR.3. Criterion: CPU parameter difference <=1e-6; GPU <=1e-4. Outcome: PASS. Limitations: GPU kernels may differ from CPU; one-round test complemented by full trajectory comparison.

```json
{
  "client_max_differences": [
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "global_max_difference": 0.0,
  "metric_differences": {
    "train_loss": 0.0,
    "train_accuracy": 0.0,
    "train_n": 0,
    "validation_loss": 0.0,
    "validation_accuracy": 0.0,
    "validation_n": 0
  },
  "manual_aggregation_error": 1.552064768373107e-08,
  "events": [
    {
      "client_id": 25,
      "round": 1,
      "pid": 845584,
      "created": 1794891431.8737278,
      "start": 1794891432.8616817,
      "finish": 1794891433.178818,
      "n_train": 94,
      "rss": 1014861824,
      "cuda_peak": 0
    },
    {
      "client_id": 67,
      "round": 1,
      "pid": 845582,
      "created": 1794891431.8679626,
      "start": 1794891432.8617043,
      "finish": 1794891433.1796577,
      "n_train": 314,
      "rss": 1012441088,
      "cuda_peak": 0
    },
    {
      "client_id": 16,
      "round": 1,
      "pid": 845585,
      "created": 1794891431.8736508,
      "start": 1794891432.861707,
      "finish": 1794891433.1752772,
      "n_train": 300,
      "rss": 1009360896,
      "cuda_peak": 0
    },
    {
      "client_id": 36,
      "round": 1,
      "pid": 845583,
      "created": 1794891431.8740501,
      "start": 1794891432.861679,
      "finish": 1794891433.1792939,
      "n_train": 182,
      "rss": 1010253824,
      "cuda_peak": 0
    }
  ]
}
```

### G_exception

Hypothesis: An incomplete or invalid round is never committed. Design: Inject failure in one of four real-data actors; other actors terminated in finally. Criterion: Explicit exception, resources returned. Outcome: PASS. Limitations: OOM and interruption are controlled injections, not real resource exhaustion.

```json
{
  "error": "RayTaskError(RuntimeError): \u001b[36mray::LocalActor.train()\u001b[39m (pid=847062, ip=<host omitted>, actor_id=2318c21d193dc9b09a1917f001000000, repr=<part1_2.core.LocalActor object at 0x7fa4877727b0>)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"part1_2/core.py\", line 35, in train\n    if fail=='exception': raise RuntimeError('Injected client exception')\n                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nRunti",
  "available_resources": {
    "memory": 67467665408.0,
    "object_store_memory": 266693741.0,
    "node:__internal_head__": 1.0,
    "CPU": 4.0,
    "node:129.105.12.180": 1.0
  }
}
```

### G_oom

Hypothesis: An incomplete or invalid round is never committed. Design: Inject failure in one of four real-data actors; other actors terminated in finally. Criterion: Explicit exception, resources returned. Outcome: PASS. Limitations: OOM and interruption are controlled injections, not real resource exhaustion.

```json
{
  "error": "RayTaskError(MemoryError): \u001b[36mray::LocalActor.train()\u001b[39m (pid=847534, ip=<host omitted>, actor_id=8abb7a7916df60867b3d087901000000, repr=<part1_2.core.LocalActor object at 0x7fa4ccaa9010>)\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"part1_2/core.py\", line 36, in train\n    if fail=='oom': raise MemoryError('Injected allocation failure; no physical memory exhaustion')\n                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
  "available_resources": {
    "memory": 67467665408.0,
    "object_store_memory": 266693741.0,
    "CPU": 4.0,
    "node:__internal_head__": 1.0,
    "node:129.105.12.180": 1.0
  }
}
```

### G_nan

Hypothesis: An incomplete or invalid round is never committed. Design: Inject failure in one of four real-data actors; other actors terminated in finally. Criterion: Explicit exception, resources returned. Outcome: PASS. Limitations: OOM and interruption are controlled injections, not real resource exhaustion.

```json
{
  "error": "FloatingPointError: Non-finite update from client 25: 1.weight",
  "available_resources": {
    "memory": 67467665408.0,
    "object_store_memory": 266258262.0,
    "CPU": 4.0,
    "node:__internal_head__": 1.0,
    "node:129.105.12.180": 1.0
  }
}
```

### G_interrupt

Hypothesis: An incomplete or invalid round is never committed. Design: Inject failure in one of four real-data actors; other actors terminated in finally. Criterion: Explicit exception, resources returned. Outcome: PASS. Limitations: OOM and interruption are controlled injections, not real resource exhaustion.

```json
{
  "error": "InterruptedError: Injected interruption: round not committed",
  "available_resources": {
    "memory": 67467665408.0,
    "object_store_memory": 268000182.0,
    "CPU": 4.0,
    "node:__internal_head__": 1.0,
    "node:129.105.12.180": 1.0
  }
}
```

### G_unavailable_GPU

Hypothesis: Unavailable GPU requests fail with bounded waiting. Design: CPU-only Ray runtime; request four fractional-GPU Actors, 0.5s deadline. Criterion: Timeout and cleanup, no aggregation. Outcome: PASS. Limitations: Tests a logical resource shortage without occupying another GPU.

```json
{
  "error": "GetTimeoutError",
  "resources": {
    "memory": 67467665408.0,
    "object_store_memory": 268435456.0,
    "node:__internal_head__": 1.0,
    "CPU": 4.0,
    "node:129.105.12.180": 1.0
  }
}
```

### G_tiny_and_recovery

Hypothesis: Tiny, missing-class clients and replay after failure preserve FedAvg. Design: Synthetic n=1,2,7,13, all class zero; then replay failed real round. Criterion: Both parameter differences <=1e-6. Outcome: PASS. Limitations: Synthetic edge clients are excluded from the 100-user experiment.

```json
{
  "tiny_error": 0.0,
  "replay_error": 0.0
}
```

### A_D_equivalence_cuda:0

Hypothesis: Fractional-GPU Actor updates match CPU SGD. Design: Four unequal real clients, same initial weights, batch order, LR .3 E3. Criterion: Max parameter error <=1e-4. Outcome: PASS. Limitations: One-round GPU test; full matched trajectory uses selected CPU strategy.

```json
{
  "client_max_differences": [
    2.9802322387695312e-08,
    5.960464477539063e-08,
    5.960464477539063e-08,
    2.9802322387695312e-08
  ],
  "global_max_difference": 2.9802322387695312e-08,
  "metric_differences": {
    "train_loss": 3.481822963280479e-08,
    "train_accuracy": 0.0,
    "train_n": 0,
    "validation_loss": 4.7631618649290886e-09,
    "validation_accuracy": 0.0,
    "validation_n": 0
  },
  "manual_aggregation_error": 1.3913331403658802e-08,
  "events": [
    {
      "client_id": 25,
      "round": 1,
      "pid": 870054,
      "created": 1794892126.7390664,
      "start": 1794892128.347878,
      "finish": 1794892128.6868494,
      "n_train": 94,
      "rss": 1707929600,
      "cuda_peak": 69411328
    },
    {
      "client_id": 67,
      "round": 1,
      "pid": 870053,
      "created": 1794892126.7127602,
      "start": 1794892128.347876,
      "finish": 1794892128.6927912,
      "n_train": 314,
      "rss": 1709445120,
      "cuda_peak": 70104576
    },
    {
      "client_id": 16,
      "round": 1,
      "pid": 870055,
      "created": 1794892126.7222881,
      "start": 1794892128.347877,
      "finish": 1794892128.7041554,
      "n_train": 300,
      "rss": 1722376192,
      "cuda_peak": 70060544
    },
    {
      "client_id": 36,
      "round": 1,
      "pid": 870056,
      "created": 1794892126.7291486,
      "start": 1794892128.3478768,
      "finish": 1794892128.6895568,
      "n_train": 182,
      "rss": 1707667456,
      "cuda_peak": 69688320
    }
  ],
  "placements": [
    {
      "pid": 870054,
      "created": 1794892126.7390664,
      "cid": 25,
      "visible": "GPU-cef65388-88b0-f39a-6bac-c5316eb78540",
      "gpu_ids": [
        "GPU-cef65388-88b0-f39a-6bac-c5316eb78540"
      ],
      "gpu": "NVIDIA GeForce RTX 5060 Ti",
      "rss": 1165864960,
      "torch_threads": 1
    },
    {
      "pid": 870053,
      "created": 1794892126.7127602,
      "cid": 67,
      "visible": "GPU-cef65388-88b0-f39a-6bac-c5316eb78540",
      "gpu_ids": [
        "GPU-cef65388-88b0-f39a-6bac-c5316eb78540"
      ],
      "gpu": "NVIDIA GeForce RTX 5060 Ti",
      "rss": 1168023552,
      "torch_threads": 1
    },
    {
      "pid": 870055,
      "created": 1794892126.7222881,
      "cid": 16,
      "visible": "GPU-cef65388-88b0-f39a-6bac-c5316eb78540",
      "gpu_ids": [
        "GPU-cef65388-88b0-f39a-6bac-c5316eb78540"
      ],
      "gpu": "NVIDIA GeForce RTX 5060 Ti",
      "rss": 1171263488,
      "torch_threads": 1
    },
    {
      "pid": 870056,
      "created": 1794892126.7291486,
      "cid": 36,
      "visible": "GPU-cef65388-88b0-f39a-6bac-c5316eb78540",
      "gpu_ids": [
        "GPU-cef65388-88b0-f39a-6bac-c5316eb78540"
      ],
      "gpu": "NVIDIA GeForce RTX 5060 Ti",
      "rss": 1166143488,
      "torch_threads": 1
    }
  ]
}
```

### F_sampling

Hypothesis: Uniform four-client sampling is reproducible. Design: 10000 rounds; two identical seeds and one different seed. Criterion: Exactly four unique IDs, same-seed equality, different-seed difference, max z<5. Outcome: PASS. Limitations: Marginal counts are a diagnostic, not proof of uniformity.

```json
{
  "min": 351,
  "max": 439,
  "expected": 400,
  "max_abs_z": 2.500520779091161,
  "counts": [
    408,
    421,
    404,
    409,
    428,
    402,
    401,
    393,
    409,
    382,
    363,
    384,
    391,
    411,
    384,
    404,
    391,
    408,
    405,
    407,
    409,
    388,
    376,
    395,
    408,
    396,
    395,
    429,
    432,
    364,
    376,
    398,
    413,
    435,
    405,
    394,
    390,
    428,
    426,
    415,
    393,
    402,
    374,
    398,
    361,
    407,
    385,
    408,
    392,
    402,
    419,
    410,
    393,
    389,
    394,
    399,
    408,
    395,
    351,
    428,
    402,
    412,
    362,
    369,
    427,
    404,
    403,
    415,
    401,
    394,
    393,
    439,
    417,
    415,
    419,
    361,
    384,
    425,
    406,
    394,
    384,
    407,
    432,
    381,
    423,
    383,
    428,
    389,
    380,
    393,
    405,
    390,
    396,
    435,
    378,
    425,
    361,
    405,
    385,
    393
  ]
}
```

### D_synthetic_aggregation

Hypothesis: Unequal counts produce manual weighted average. Design: Reuse Part 1.1 deterministic vectors. Criterion: Error <=1e-6. Outcome: PASS. Limitations: Small synthetic arithmetic test.

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

### B/E: actual concurrency and lifecycle

Hypothesis: selected clients run in separate overlapping processes and release all resources. Design: inspect measured start/finish timestamps, four PIDs per round, CPU resources after cleanup, and driver RSS across all 80 rounds. Criterion: observed overlap, exactly four Actors per round, zero surviving Actor PIDs and four returned CPUs. Results:
```json
{
  "rounds": 80,
  "client_actors_per_round": 4,
  "distinct_processes_per_round": true,
  "maximum_training_concurrency": 4,
  "rounds_with_four_overlapping": 80,
  "all_actor_processes_exited_between_rounds": true,
  "all_cpu_resources_returned": true,
  "driver_rss_first_MiB": 1171.48828125,
  "driver_rss_last_MiB": 1172.921875,
  "driver_rss_max_MiB": 1172.921875,
  "mean_fastest_slowest_gap_seconds": 0.0329291969537735,
  "maximum_fastest_slowest_gap_seconds": 0.14037346839904785,
  "mean_client_training_seconds": 0.3168298177421093,
  "mean_actor_startup_seconds": 2.6881112146482335,
  "mean_cleanup_seconds": 0.09368039273540489,
  "mean_aggregation_seconds": 0.0011728106590453535,
  "mean_completion_wait_seconds": 0.01268317773938179
}
```

The timeline displays actual timestamps, including Actor initialization and aggregation. Timing differences in local dataset sizes and computation make synchronous FedAvg wait for the slowest client; the reported fastest/slowest gap quantifies one aspect of this straggler effect. RSS is observational evidence and is not a formal leak detector; process exit and returned resources provide stronger lifecycle evidence.

### Additional robustness and repeatability

Controlled tests distinguish injected MemoryError from actual hardware exhaustion. Exceptions, nonfinite updates and interrupted rounds are reported, all Actors cleaned up, and no partial aggregate committed. Extra tests use tiny synthetic clients, a CPU-only cluster requesting unavailable GPU resources with bounded wait, and deterministic replay after interruption. Results:
```json
{
  "unavailable_gpu": {
    "reported": "GetTimeoutError",
    "cpus_returned": 4.0
  },
  "tiny_missing_classes": {
    "sizes": [
      1,
      2,
      7,
      13
    ],
    "parameter_difference": 0.0
  },
  "replay_and_repeat": {
    "parameter_difference": 0.0,
    "input_unchanged": true
  },
  "passed": true
}
```

## Limitations and three-month extension plan

Only one full training seed was used; fixed-seed repeatability is separate from across-seed statistical stability. GPU numerical evidence covers one round; the chosen strategy has the full matched 80-round comparison. OOM is injected safely, not produced by exhausting physical RAM. A deliberately killed driver and actual machine/network failure were not tested; interruption is injected before commit and restart logic uses atomic checkpoints. Empty clients and invalid pixel/label cases are covered by preserved Part 1.1 tests, not observed in this real dataset.

Month 1: repeat timing over 10+ independent Ray sessions and training over 5-10 seeds, bootstrap at client level, vary client sizes/model size and quantify crossover workloads. Month 2: inject real worker termination and driver restart in isolated jobs, test bounded retries, heterogeneous delays, memory pressure under safe quotas and numerical agreement across architectures. Month 3: prespecify CPU/GPU budgets, collect CPU-seconds/energy/communication and client-fairness metrics, compare fixed-compute budgets, run long lifecycle stress tests and validate on newly untouched writer-held-out data. Do not repeatedly tune against the current test set.

## Figures and reproduction

Five 300-DPI PNG/vector PDF pairs: loss, accuracy, matched sequential/Ray curves, performance, and actual Actor execution timeline. See the [component README](../part1_2/README.md) for exact commands. The completed [Part 2](../part2/README.md) and [combined report](../report/README.md) provide the remaining assignment material.

Publication note: personal paths and host addresses in these Markdown excerpts are omitted. Original numerical JSON/CSV evidence is unchanged. Generated notes can restore machine metadata; consult the component README before regeneration or publication.
