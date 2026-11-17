# Dataset setup

Federated EMNIST course dataset (local input directory)

Purpose and access
------------------
This directory holds the instructor-supplied Assignment 3 dataset, adapted from
LEAF. Obtain the files through the course distribution; the project does not
download them. Redistribution permission has not been established. Keep the raw
.npy files, and Ray's derived client .npz arrays, out of the public repository.
This README contains no example images, labels, or raw data.

Required layout, relative to the repository root:
  Assignment3-data/train_data.npy
  Assignment3-data/test_data.npy

The trusted NumPy object arrays require allow_pickle=True. Only load trusted
course files. train_data contains 100 client dictionaries indexed by client ID;
each has images and labels. test_data[0] contains the aggregate held-out images
and labels. Images are 28x28 normalized grayscale; labels are numeric IDs 0-61.
No verified character-order lookup is supplied.

Counts and preprocessing
------------------------
Development: 31,825 examples across 100 clients and 62 classes.
Saved client split: 25,418 training and 6,407 validation examples.
Held-out test: 3,621 examples in the separate test_data.npy file.
The implementation uses 1-X and int64 labels. The reproducible split seed is 490;
reuse part1_1_deepdish5_outputs/splits.json across all components.

Inspection and validation
-------------------------
From the repository root, in an environment with Python, PyTorch, NumPy,
Matplotlib, and psutil (recorded versions are in ../README.md):
  python -m part1_1 inspect --device cpu
  python -m unittest discover -s part1_1/tests -v

Inspection writes class analysis under part1_1_deepdish5_outputs/analysis/ and
creates/reuses the split manifest. The correctness suite uses synthetic fixtures;
scientific controls and full training are documented in ../part1_1/README.md,
../part1_2/README.md, and ../part2/README.md. Those commands should run in a
separate reproduction copy to preserve completed outputs. They never require
editing the original dataset. Plot regeneration from completed Part 1.1 results:
  python -m part1_1 plots

The Ray prepare phase writes dataset-derived training shards to
part1_2_outputs/clients/. CPU inspection is supported; the original training used
DeepDish5 with compatible CUDA/PyTorch hardware. This directory has no standalone
training implementation or installation script.

Reference
---------
Caldas et al., LEAF: A Benchmark for Federated Settings, 2018.
https://arxiv.org/abs/1812.01097
