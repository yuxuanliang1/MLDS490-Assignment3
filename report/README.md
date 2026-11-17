# Assignment 3 report: Overleaf source

[Project overview](../README.md) | [Main LaTeX source](main.tex) | [Bibliography](references.bib)

## Purpose and structure

This is the completed academic report for sequential FedAvg, Ray parallel clients,
and Laplace input perturbation. It contains required class/convergence/comparison
plots, scientific controls, edge cases, three non-standard plots, reproduction
instructions, and a proposed three-month validation plan.

- `main.tex`: title page, packages, section order, and float boundaries.
- `sections/`: scientific narrative, numerical tables, reproduction appendix.
- `figures/`: 20 packaged assets (11 existing PDF plots and 9 PNG plots).
- `references.bib`: seven research/course bibliography entries.
- `check_report.py`: standard-library-only static consistency checker.
- `tools/regenerate_figures.py`: optional report-sized PNG regeneration.

There are 20 numbered figures and 14 tables. Numerical tables are LaTeX text,
not images. The package is self-contained for compilation: it needs no raw data,
model checkpoint, Python environment, shell escape, or custom font.

## Upload to Overleaf

Upload the current directory contents, preserving `sections/` and `figures/`.
Set `main.tex` as the main file and select pdfLaTeX with a current TeX Live.
Overleaf handles BibTeX and additional reference passes. Existing workspace ZIPs
may predate these edits; use the current folder. The final compiled report is available at the repository root. The user compiles and visually checks the report in Overleaf; no local
compilation is part of this workflow.

The cover page retains Yuxuan Liang and the correct GitHub identifiers. Section 8
prints the full repository URL using a breakable `url` command.

## Static checks

From the repository root, with standard-library Python:

```bash
python report/check_report.py
```

This checks source/figure paths, table columns, labels, citations, structural
syntax, figure hashes, and required sections. It writes `STATIC_AUDIT.json` and
does not compile LaTeX or execute models. Historical numerical and revision audit
JSON files record prior checks; they are not included in the typeset report.
[Source audit notes](SOURCE_AUDIT.md) distinguish the earlier revisions from the
current layout-only changes. No source checker can certify final pagination.

## Figure regeneration and experiment reproduction

With NumPy and Matplotlib installed and the completed output directories present:

```bash
python report/tools/regenerate_figures.py
```

This optional command reads saved counts, histories, events, and client statistics
to regenerate nine report-sized PNGs and their manifest hashes. It does not train
models or create a report PDF. To regenerate the original component plots, use
the commands in [Part 1.1](../part1_1/README.md), [Part 1.2](../part1_2/README.md),
and [Part 2](../part2/README.md). Part 2's image panel also requires the local
course development images. Those guides provide dependencies, dataset placement,
correctness tests, full scientific validation, and experiment execution order.
Overleaf itself does not require the experiment environment or a GPU.

## Visual review after compilation

- Table 5 must finish before the Part 1.2 heading; the boundary has a FloatBarrier.
- Section 6 cites both headline and validation-summary tables. The summary table
  follows the complete systems/privacy paragraph, which is kept together.
- Figure 20 belongs to image diagnostics and must precede Appendix B.1; a local
  FloatBarrier separates the image figures from the synthetic control.
- Check the Section 8 URL wraps, tables remain legible, and the barriers do not
  create excessive whitespace. Confirm all references and bibliography dates.

These placements have been fixed in source but require Overleaf visual
confirmation. The user reported about 29 pages for the earlier compilation.

## Publication

Report sources and numerical plots can be distributed as a source package subject
to applicable course permissions. The image-example panel contains course-data
examples and needs redistribution review. Raw datasets and checkpoints are not
part of this folder. Auxiliary audit files are maintainer documentation; review
historical path metadata before including them in a public release.
