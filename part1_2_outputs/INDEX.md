# Part 1.2 results

[Report notes](REPORT_NOTES.md) | [README](../part1_2/README.md) | [Final audit](final_audit.json)

[Completion summary](completion.json) | [Scientific validation](../public_evidence/part1_2_outputs/validation.json) | [Extra edge cases](extra_validation.json) | [Benchmark](performance_summary.csv) | [Concurrency and lifecycle](lifecycle_concurrency_audit.json) | [Final test](test.json) | [Best checkpoint](parallel/best.pt)

[Figure manifest](figures/manifest.json). Five PNG/PDF figure pairs are in figures/.

## Completed results and protocol

The four-CPU-Actor run selects round 78 by validation and scores 73.79% on the
held-out set. Matched 80-round sequential/Ray trajectories have maximum parameter
and metric difference 0.0. All 80 rounds have four overlapping training intervals
and successful lifecycle cleanup. Full workloads take 276.14 s Ray versus 11.95 s
sequential, approximately 23.1x slower for this small model.

## Structure, dependencies, and reproduction

`parallel/` and `sequential/` hold histories, selections, events/timings, summaries,
and checkpoints; `clients/` holds generated training arrays. The
[component guide](../part1_2/README.md) documents required Python/PyTorch/Ray/psutil
versions, the instructor data and shared split manifest, Actor allocations,
correctness/failure checks, scientific validation, and full reproduction order.
Use one thread per process and a separate reproduction copy. From the root,
`python -m part1_2.workflow artifacts` regenerates the five figure pairs and
measured lifecycle summary without training or held-out scoring.

`clients/*.npz` files are dataset-derived and should remain private. Checkpoints,
raw exception traces, host logs, and environment records need publication review.
Generated notes may reproduce historical paths; review them before publication.
The completed Part 2 and final report are linked from the [project overview](../README.md).
