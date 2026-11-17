# Report source verification

This file is internal package documentation and is not input by `main.tex`.

## Current layout and documentation revision

The current changes add a float barrier before Part 1.2, explicitly cite both
Section 6 tables, place the validation summary after the complete systems/privacy
paragraph, and settle image diagnostics before Appendix B.1. The synthetic ceiling
statement is concise, and Section 8 prints the full breakable GitHub URL. The cover
page is unchanged. Table bodies, figure assets, experimental code, and numerical
CSV/JSON/checkpoint records are preserved.

Project README/INDEX documentation now describes the completed workflows and
publication boundaries. Generated Markdown notes have obsolete status text and
personal paths removed; underlying experiment records retain their historical
bytes. Historical full-tree audits may include older documentation hashes and
should not be rewritten to conceal subsequent documentation changes.

The following sections describe the earlier broad report revision, not a new
experiment or a claim of current compiled pagination.

## Scope and preservation

The existing source was revised directly, retaining the user's title page, author, submission date, and repository identifiers. The original report was backed up to `tmp/report_revision/baseline.zip` before editing. No training, inference, new experiment, local TeX compilation, or report PDF generation was performed.

All 13 retained numerical tables have identical numerical tokens to the baseline. The former question-coverage table was replaced by a concise scientific-validation summary, yielding 14 tables overall. All 21 seed-specific noise-result rows remain. A SHA-256 comparison of 1,007 implementation, dataset, result, and checkpoint files found no changes. `REVISION_AUDIT.json` records these checks.

The same source-word proxy (alphabetic tokens after stripping LaTeX commands and code listings) decreased from 11,152 to approximately 7,350, about one third. This proxy includes captions/tables and is not a typeset word count.

## Source and scientific checks

- Reviewed the assignment handout, supplied Week 7 lecture/lab materials, existing sources, experiment notes, result summaries, and available source ZIP.
- Preserved the FedAvg equation and centralized-SGD identity; positive/negative controls; sampling and edge-case evidence; Ray exact equivalence, overlap, lifecycle, timings and failure injection; Laplace moments, quantiles, independence diagnostics, exact zero-noise recovery, all seven scales and three seeds, noise-reuse ablation, stress tests, and non-standard plots.
- Preserved distinctions among selected checkpoints, final-round outcomes, means, sample SDs, and held-out test protocols.
- Corrected the selection terminology and added the explicit validation-accuracy condition. Cross-entropy loss references remain where appropriate.
- Retained the detailed formal-privacy limitation and a measurable three-month proposed validation plan.
- Checked displayed command phases and options against current CLI implementations. Complete ordered workflows are in Appendix D; machine-specific launchers are described briefly.
- The existing read-only numerical verifier passed 154 checks using saved records, CSV trajectories, selections, and file digests. It does not execute model training or evaluation. Parameter-level experimental results remain attributed to the original validation records.

## Figures and layout

Nine PNG figures were regenerated exclusively from saved counts, histories, events, and client statistics. They replace ten report figure PDFs because the Ray accuracy and loss panels are combined into one readable figure. The original experimental PDFs outside `report/` remain unchanged. `FIGURE_MANIFEST.json` records each asset's hash and either original source or numerical-source hashes.

The overall and five-client distributions retain all 62 bars, with spaced numeric ticks. The heterogeneity and Actor timeline captions describe only the displayed comparison/round; omitted controls and all-round overlap retain numerical discussion. The robustness heatmap retains every client/scale cell and marks the three largest losses. All nine PNGs were visually reviewed for labels, legends, clipping, and aspect ratio. Existing numerical tables remain selectable LaTeX content.

Reduced float spacing, flexible placement, and fewer forced barriers target unnecessary whitespace. Supporting perturbation plots moved to appendices. These are source-level improvements; final pagination and overflow must be checked in Overleaf.

## Static validation

`python report/check_report.py` originally passed 355 checks; the latest static audit passes 362 checks covering source dependencies, figure inventory/hashes, table column counts, brace/math/environment balance, unique/resolved labels, all seven cited bibliography entries, required sections and plots, correct repository identifiers, and absence of the flagged production phrases and personal server path. The checker never invokes a compiler.

No compiled report PDF was present in the workspace. The user reported approximately 29 pages for the previous Overleaf compilation; the latest layout changes require a fresh visual check there. The package is self-contained for compilation; optional plot regeneration depends on the separate saved experiment outputs.
