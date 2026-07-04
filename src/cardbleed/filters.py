"""Small separable-convolution helpers (pure NumPy, no SciPy)."""

from __future__ import annotations

import math

import numpy as np


def conv_axis(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve one axis of `a` with a 1-D kernel (reflect padding)."""
    r = len(kernel) // 2
    if a.shape[axis] < 2:
        return a.astype(np.float32, copy=True)
    pads = [(0, 0)] * a.ndim
    pads[axis] = (r, r)
    # reflect needs pad < dim; fall back to edge for tiny arrays
    mode = "reflect" if a.shape[axis] > r else "edge"
    ap = np.pad(a, pads, mode=mode)
    out = np.zeros(a.shape, dtype=np.float32)
    sl = [slice(None)] * a.ndim
    for i, kv in enumerate(kernel):
        sl[axis] = slice(i, i + a.shape[axis])
        out += np.float32(kv) * ap[tuple(sl)]
    return out


def gauss_kernel(sigma: float) -> np.ndarray:
    r = max(1, math.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def gaussian_blur2d(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur over the first two axes of (H,W) or (H,W,C)."""
    if sigma <= 0:
        return a.astype(np.float32, copy=True)
    k = gauss_kernel(sigma)
    return conv_axis(conv_axis(a, k, 0), k, 1)


def box_blur3(a: np.ndarray) -> np.ndarray:
    k = np.full(3, 1 / 3, dtype=np.float32)
    return conv_axis(conv_axis(a, k, 0), k, 1)


def highpass_std(region: np.ndarray) -> np.ndarray:
    """Per-channel std of the high-frequency residual of an (H,W,C) region."""
    return (region.astype(np.float32) - box_blur3(region)).std(axis=(0, 1))


def smooth_field(
    rng: np.random.Generator,
    shape: tuple[int, int],
    sigma: float,
    *,
    standardize: bool = False,
) -> np.ndarray:
    """Random field blurred to a given correlation length.

    Standardized fields have mean 0 / std 1 (for signed displacements);
    otherwise the field is renormalized to span [0, 1].
    """
    f = gaussian_blur2d(rng.random(shape, dtype=np.float32), sigma)
    if standardize:
        return (f - f.mean()) / max(float(f.std()), 1e-6)
    return (f - f.min()) / max(float(np.ptp(f)), 1e-6)
