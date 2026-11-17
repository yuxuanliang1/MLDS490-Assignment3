# Assignment 3 Part 2 artifacts

[Report notes](REPORT_NOTES.md) | [README](../part2/README.md) | [Final audit](final_audit.json)

- [All 21 final-round results](final_results.csv)
- [Across-seed means and sample SD](seed_summary.csv)
- [Frozen validation-based recommendation and scoring protocol](frozen_protocol.json)
- [Scientific validation A-J and edge cases](validation/results.json)
- [All test scores and separately labeled Question1 checkpoint](test_results.json)
- [Environment and GPU UUID](../public_evidence/part2_outputs/environment.json)
- Preservation checksums for prior work (`protected_snapshot.json`, retained locally)

| Figure | PNG | PDF |
|---|---|---|
| dp_accuracy_convergence | [PNG](figures/dp_accuracy_convergence.png) | [PDF](figures/dp_accuracy_convergence.pdf) |
| dp_loss_convergence | [PNG](figures/dp_loss_convergence.png) | [PDF](figures/dp_loss_convergence.pdf) |
| dp_accuracy_vs_noise_scale | [PNG](figures/dp_accuracy_vs_noise_scale.png) | [PDF](figures/dp_accuracy_vs_noise_scale.pdf) |
| dp_noise_seed_stability | [PNG](figures/dp_noise_seed_stability.png) | [PDF](figures/dp_noise_seed_stability.pdf) |
| dp_noise_distribution_validation | [PNG](figures/dp_noise_distribution_validation.png) | [PDF](figures/dp_noise_distribution_validation.pdf) |
| dp_image_perturbations | [PNG](figures/dp_image_perturbations.png) | [PDF](figures/dp_image_perturbations.pdf) |
| dp_client_robustness_heatmap | [PNG](figures/dp_client_robustness_heatmap.png) | [PDF](figures/dp_client_robustness_heatmap.pdf) |
| dp_privacy_utility_tradeoff | [PNG](figures/dp_privacy_utility_tradeoff.png) | [PDF](figures/dp_privacy_utility_tradeoff.pdf) |

Individual configurations, per-round CSV/JSON histories, noise hashes/diagnostics, selections, initial/best/resumable final checkpoints and per-client final validation metrics are in experiments/.

Baseline/recommendation use validation only. Test comparisons always use fixed final-round checkpoints. No formal epsilon-DP guarantee is claimed.

## Completed result and reproduction

The predefined utility-preserving choice is b=0.05: 71.80% mean final validation
accuracy and 72.66% mean round-80 test accuracy over three seeds. The zero-noise
mean test accuracy is 73.38% +/- 0.66 percentage points. The separate Question 1
b=0.10/seed-2026 selected checkpoint scores 71.64%; it is not a three-seed mean.
No verified end-to-end epsilon-DP guarantee is established.

The [component guide](../part2/README.md) specifies dependencies, input placement,
implementation modules, mathematical/correctness checks, scientific controls,
and the complete init-to-frozen-test command order. Full zero-noise validation
requires the completed Part 1.1 baselines. Part 2 also snapshots prior component
files, so complete prerequisite work before initialization in a fresh reproduction
copy. Keep historical snapshots unchanged after documentation revisions.

From the repository root, `python -m part2.workflow artifacts` regenerates all
eight plot pairs and notes from completed outputs; the image panel additionally
reads the development data. This performs no training or held-out scoring.
Use the recorded DeepDish5/PyTorch environment for matched GPU reproduction;
CPU mode is documented in the component guide. Raw course files, checkpoints,
image-example figures, and environment/preservation records require separate
publication review. The [report guide](../report/README.md) covers Overleaf.
