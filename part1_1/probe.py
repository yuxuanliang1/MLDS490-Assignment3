"""Read-only environment and compatibility probe, executed on DeepDish5."""
import importlib.util
import os
import platform
import sys

print("Python", sys.executable, sys.version, "CPU", platform.processor(), os.cpu_count(), flush=True)
for module in ["torch", "numpy", "matplotlib", "psutil"]:
    print(module, bool(importlib.util.find_spec(module)), flush=True)
import torch
print("PyTorch", torch.__version__, "CUDA build", torch.version.cuda,
      "CUDA available", torch.cuda.is_available(), "compiled", torch.cuda.get_arch_list(), flush=True)
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_properties(0), flush=True)
    x = torch.randn(32, 784, device="cuda")
    w = torch.randn(784, 128, device="cuda", requires_grad=True)
    (x @ w).square().mean().backward()
    torch.cuda.synchronize()
    print("CUDA float32 forward/backward succeeded", flush=True)
