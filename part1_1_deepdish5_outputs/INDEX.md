# Part 1.1 completed artifacts

Completed final experiments ran on DeepDish5. This directory is the Part 1.1 evidence archive, not a separate implementation.

**Validation-selected seed-2026/round-80 test accuracy: 74.12% (2,684/3,621).** The same configuration at fixed round 80 gives 73.38% +/- 0.66 percentage points across three seeds, using the matched Part 2 zero-noise results on the same test set.

- [Report notes with actual numerical results](REPORT_NOTES.md)
- [Setup, exact reproduction commands and implementation guide](../part1_1/README.md)
- [Six-setting grid plus four seed repeats](comparison.csv)
- [Scientific validation: hypotheses, numerical results and limitations](validation/results.json)
- Correctness test log (`validation/correctness_tests.log`, retained locally)
- [Frozen validation-based selection](selection.json)
- [Final test evaluation](final_test.json)
- [Selected trained checkpoint](experiments/c10_e3_s2026/best.pt)
- [Final artifact audit](final_audit.json)
- [Hardware/environment](../public_evidence/part1_1_deepdish5_outputs/environment.json)
- [Package versions](package_versions.txt)
- Source and supplied-input SHA-256 manifest (`source_manifest.json`, retained locally)

## Figures

Each figure is saved as a 300-DPI PNG and vector PDF.

| Figure | PNG | PDF |
|---|---|---|
| overall class distribution | [PNG](figures/overall_class_distribution.png) | [PDF](figures/overall_class_distribution.pdf) |
| five client distributions | [PNG](figures/five_client_distributions.png) | [PDF](figures/five_client_distributions.pdf) |
| client class heatmap | [PNG](figures/client_class_heatmap.png) | [PDF](figures/client_class_heatmap.pdf) |
| baseline convergence | [PNG](figures/baseline_convergence.png) | [PDF](figures/baseline_convergence.pdf) |
| hyperparameter accuracy | [PNG](figures/hyperparameter_accuracy.png) | [PDF](figures/hyperparameter_accuracy.pdf) |
| accuracy vs client epochs | [PNG](figures/accuracy_vs_client_epochs.png) | [PDF](figures/accuracy_vs_client_epochs.pdf) |
| seed stability | [PNG](figures/seed_stability.png) | [PDF](figures/seed_stability.pdf) |
| participation heterogeneity benefit | [PNG](figures/participation_heterogeneity_benefit.png) | [PDF](figures/participation_heterogeneity_benefit.pdf) |
| participation histogram | [PNG](figures/participation_histogram.png) | [PDF](figures/participation_histogram.pdf) |
| scientific controls | [PNG](figures/scientific_controls.png) | [PDF](figures/scientific_controls.pdf) |
| pilot learning rates | [PNG](figures/pilot_learning_rates.png) | [PDF](figures/pilot_learning_rates.pdf) |

Earlier local pilots in `../part1_1_outputs/` are excluded from these final results. The completed Ray and Laplace components and combined report are linked from the [project overview](../README.md).

## Reproduction and publication

Dependencies, dataset placement, CLI options, correctness tests, scientific controls,
and the ordered selection/test workflow are in the [component guide](../part1_1/README.md).
Use a separate reproduction copy. To regenerate these plots from completed results,
run `python -m part1_1 plots` from the repository root; no training or held-out
scoring is involved. Per-run histories/configurations live in `experiments/`;
`analysis/` contains saved counts and `validation/` contains scientific evidence.

The recorded environment is DeepDish5 with the versions in the root README.
Checkpoint, machine-log, environment, and provenance links above identify local
artifacts; they are not recommendations to commit them publicly. Keep raw data
private and review the selected publication subset. Historical source hashes
include earlier documentation bytes; do not alter their records after doc edits.
