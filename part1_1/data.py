"""Trusted supplied NumPy object file; fixed per-user disjoint splits.

No training/analysis code opens test_data.npy. Only final_evaluation does so.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .utils import keyed_seed, read_json, sha256, write_csv, write_json


def split_indices(n, seed):
    """floor(.8*n) train, remainder validation; n=1: train only; n=0: empty."""
    if n < 0:
        raise ValueError("Negative client size")
    order = np.random.default_rng(seed).permutation(n)
    cut = min(n - 1, max(1, int(0.8 * n))) if n >= 2 else n
    return order[:cut], order[cut:]


def validate_arrays(images, labels):
    x, y = np.asarray(images), np.asarray(labels)
    # Empty Python lists have no shape/dtype information. Canonicalize their
    # zero-example representation without introducing any examples or labels.
    if y.ndim == 1 and len(y) == 0 and x.size == 0:
        return np.empty((0, 28, 28), dtype=np.float32), np.empty(0, dtype=np.int64)
    if x.shape != (len(y), 28, 28) or y.ndim != 1:
        raise ValueError(f"Unexpected image/label shape: {x.shape}, {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Labels must be integer class indices")
    if not np.isfinite(x).all() or (len(y) and (x.min() < 0 or x.max() > 1)):
        raise ValueError("Expected finite supplied pixels in [0,1]; no silent rescaling")
    if len(y) and (y.min() < 0 or y.max() >= 62):
        raise ValueError("Label outside [0,61]")
    return x.astype(np.float32, copy=False), y.astype(np.int64, copy=False)


@dataclass
class ClientData:
    client_id: int
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor

    @property
    def n_train(self):
        return len(self.train_y)


def load_development(data_dir, manifest_path, device="cpu", split_seed=490):
    path = Path(data_dir) / "train_data.npy"
    source_hash = sha256(path)
    raw = np.load(path, allow_pickle=True)  # Instructor-provided trusted object file.
    if raw.dtype != object or raw.ndim != 1:
        raise ValueError("Expected one-dimensional array of client dictionaries")
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["source_sha256"] != source_hash or manifest["split_seed"] != split_seed:
            raise ValueError("Split manifest source/seed mismatch; use a distinct output directory")
    else:
        manifest = dict(source_sha256=source_hash, split_seed=split_seed,
                        policy="floor(0.8*n), n>=2: both nonempty; singleton train only; empty ineligible",
                        clients={})
        for cid, item in enumerate(raw):
            tr, va = split_indices(len(item["labels"]), keyed_seed(split_seed, cid))
            manifest["clients"][str(cid)] = dict(train=tr.tolist(), validation=va.tolist())
        write_json(manifest_path, manifest)
    if len(manifest["clients"]) != len(raw):
        raise ValueError("Client count mismatch")
    clients = {}
    for cid, item in enumerate(raw):
        if set(item) != {"images", "labels"}:
            raise ValueError(f"Unexpected keys on client {cid}: {list(item)}")
        x, y = validate_arrays(item["images"], item["labels"])
        # Supplied glyphs are dark on an almost-white background. Put ink at 1
        # and background at 0 without estimating anything from validation/test.
        x = 1.0 - x
        m = manifest["clients"][str(cid)]
        tr, va = np.asarray(m["train"], dtype=int), np.asarray(m["validation"], dtype=int)
        union = np.concatenate([tr, va])
        if len(union) != len(y) or not np.array_equal(np.sort(union), np.arange(len(y))):
            raise ValueError(f"Split coverage, duplicates or overlap on client {cid}")
        clients[cid] = ClientData(cid, torch.as_tensor(x[tr], device=device),
                                  torch.as_tensor(y[tr], device=device),
                                  torch.as_tensor(x[va], device=device),
                                  torch.as_tensor(y[va], device=device))
    return clients, manifest


def label_statistics(y):
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=62)
    n = int(counts.sum())
    p = counts[counts > 0] / n if n else np.array([])
    return dict(n=n, observed_classes=int(len(p)), dominant_proportion=float(p.max()) if n else None,
                entropy_bits=float(-(p * np.log2(p)).sum()) if n else None), counts


def analyze(clients, output, manifest):
    output = Path(output)
    rows, pool_counts, train_counts, histograms = [], np.zeros(62, int), np.zeros(62, int), []
    for cid, c in clients.items():
        ty, vy = c.train_y.cpu().numpy(), c.val_y.cpu().numpy()
        stats, counts = label_statistics(ty)
        pool_stats, combined = label_statistics(np.concatenate([ty, vy]))
        rows.append(dict(client_id=cid, pool_n=pool_stats["n"], train_n=stats.pop("n"),
                         validation_n=len(vy), **stats))
        pool_counts += combined
        train_counts += counts
        histograms.append(counts.tolist())
    chosen, used = [], set()
    for reason, key, reverse in [("largest training set", "train_n", True),
                                 ("smallest training set", "train_n", False),
                                 ("most observed training classes", "observed_classes", True),
                                 ("fewest observed training classes", "observed_classes", False)]:
        row = next(r for r in sorted(rows, key=lambda r: (-r[key] if reverse else r[key], r["client_id"])) if r["client_id"] not in used)
        used.add(row["client_id"])
        chosen.append(dict(reason=reason, **row))
    # Typical: minimum standardized squared distance to medians of size/diversity/entropy.
    fields = ["train_n", "observed_classes", "entropy_bits"]
    arr = np.array([[r[f] for f in fields] for r in rows])
    distance = (((arr - np.median(arr, axis=0)) / np.maximum(arr.std(axis=0), 1e-9)) ** 2).sum(1)
    row = next(rows[i] for i in np.argsort(distance, kind="stable") if rows[i]["client_id"] not in used)
    chosen.append(dict(reason="closest to median size, diversity and entropy", **row))
    sizes = [r["pool_n"] for r in rows]
    summary = dict(clients=len(clients), eligible_clients=sum(c.n_train > 0 for c in clients.values()),
                   pool_examples=int(pool_counts.sum()), training_examples=int(train_counts.sum()),
                   validation_examples=sum(len(c.val_y) for c in clients.values()),
                   classes=int((pool_counts > 0).sum()), label_ids=np.flatnonzero(pool_counts).tolist(),
                   image_shape=[28, 28], source_image_dtype="float32", source_label_dtype="int32",
                   normalization="1 - x: supplied dark ink / white background becomes positive ink / zero background; no rotations or transposes",
                   label_mapping="Only numeric IDs documented in supplied files; no character mapping assumed",
                   size_statistics=dict(min=min(sizes), max=max(sizes), mean=float(np.mean(sizes)),
                                        median=float(np.median(sizes)),
                                        quartiles=np.quantile(sizes, [.25, .75]).tolist()),
                   representative_clients=chosen, source_sha256=manifest["source_sha256"])
    write_json(output / "dataset_summary.json", summary)
    write_csv(output / "client_statistics.csv", rows)
    write_json(output / "class_counts.json", dict(development_pool=pool_counts.tolist(),
               training=train_counts.tolist(), client_training=histograms))
    return summary
