"""Small reproducibility and artifact helpers; no federated framework dependencies."""
import csv
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def keyed_seed(*keys):
    return int.from_bytes(hashlib.sha256(":".join(map(str, keys)).encode()).digest()[:4], "little")


def setup(seed=2026, threads=2):
    # Set before CUDA initialization. TF32 is disabled for interpretable float32 checks.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_device(name):
    if name == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu explicitly")
    return name


def environment(device):
    import matplotlib
    import psutil
    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        smi = str(error)
    result = dict(hostname=socket.gethostname(), platform=platform.platform(),
                  python=sys.version, executable=sys.executable, torch=torch.__version__,
                  numpy=np.__version__, matplotlib=matplotlib.__version__,
                  cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                  cuda_architectures=torch.cuda.get_arch_list(), device=device,
                  logical_cpus=os.cpu_count(), cpu=platform.processor(),
                  ram_bytes=psutil.virtual_memory().total,
                  ram_available_bytes=psutil.virtual_memory().available,
                  threads=torch.get_num_threads(), nvidia_smi=smi)
    if device.startswith("cuda"):
        p = torch.cuda.get_device_properties(device)
        result.update(gpu=p.name, gpu_memory_bytes=p.total_memory,
                      gpu_compute_capability=list(torch.cuda.get_device_capability(device)))
    return result


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path):
    # Only load artifacts produced locally by this code, never untrusted checkpoints.
    return torch.load(path, map_location="cpu", weights_only=False)
