"""Noise is sampled with NumPy on CPU and transferred unchanged to the device.

This makes CPU/CUDA noise identical for a given key and avoids global RNG state.
Network inputs are 1-X, so perturbing raw X by +epsilon gives (1-X)-epsilon.
"""
import numpy as np
import torch
from part1_1.data import ClientData
from part1_1.utils import keyed_seed


def laplace_noise(shape, b, seed, device='cpu'):
    if not np.isfinite(b) or b < 0:
        raise ValueError('Laplace scale must be finite and nonnegative')
    if b == 0:
        return torch.zeros(shape, dtype=torch.float32, device=device)
    with np.errstate(over='ignore', invalid='ignore'):
        array = np.random.default_rng(seed).laplace(0., b, size=shape).astype(np.float32)
    if not np.isfinite(array).all():
        raise FloatingPointError('Laplace sample exceeds finite float32 range')
    return torch.as_tensor(array, device=device)


def perturb(x, b, seed):
    if x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError('Expected finite float32 images')
    if not np.isfinite(b) or b < 0:
        raise ValueError('Laplace scale must be finite and nonnegative')
    if b == 0:
        return x.clone()  # No random generator is called.
    result = x - laplace_noise(tuple(x.shape), b, seed, str(x.device))
    if not torch.isfinite(result).all():
        raise FloatingPointError('Non-finite perturbed images')
    return result


def noise_key(seed, cid, b, *extra):
    return keyed_seed(seed, 'laplace', float(b).hex(), cid, *extra)


def perturbed_clients(clean, b, seed):
    return {cid: ClientData(cid, perturb(c.train_x, b, noise_key(seed, cid, b)),
             c.train_y, c.val_x, c.val_y) for cid, c in clean.items()}
