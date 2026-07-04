"""Border synthesis: edge analysis and pattern-continuing extension.

All four edges reduce to one canonical operation ("extend the LEFT edge") via
np.rot90, so there is a single tested code path. The smart mode splits the
border into a smooth *tone* (continued outward mirrored — ordered and
seam-continuous) and a speckle *residual* (resampled with randomized depth,
along-edge wobble, and long-range shuffle), then adds measured-grain noise and
a ramped smudge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import FileError
from .filters import (conv_axis, gauss_kernel, gaussian_blur2d, highpass_std,
                      smooth_field)

TRIM_CAP = 3          # max lines auto-trim may cut
BLOOM_STEP = 12.0     # luminance step marking a scanner bloom line
INNER_STEP = 15.0     # luminance step marking the border's inner boundary
TONE_SIGMA = 6.0      # along-edge smoothing that separates tone from speckle
SHUFFLE_SMOOTH = 4.0  # patch coherence of the long-range shuffle field

_SIDE_ROT = {"left": 0, "top": 1, "right": 2, "bottom": 3}  # CCW turns to face left


@dataclass
class Params:
    mode: str = "smart"
    sample: int = 8
    trim: str = "auto"          # "auto" or an integer as string
    jitter: float = 0.85
    jitter_smooth: float = 1.2
    jitter_cross: float = 4.0
    shuffle: float = 48.0
    noise: float = 0.35
    smudge: float = 0.6
    seam_feather: int = 3
    corner_guard: int = 12
    halo: str = "auto"          # auto | overwrite | blend


def edge_geometry(img: np.ndarray, p: Params) -> tuple[int, int, np.ndarray]:
    """Analyze the LEFT edge of img: how much scanner bloom to trim and how
    deep the usable border band is.

    Returns (t, K, profile): trim, band depth, and the median luminance
    profile that informed them.
    """
    H, W, C = img.shape
    depth_scan = min(20, W)
    if C >= 3:
        lum = img[:, :depth_scan, :3] @ np.array([0.299, 0.587, 0.114],
                                                 dtype=np.float32)
    else:  # gray / gray+alpha: the first channel IS the luminance
        lum = img[:, :depth_scan, 0].astype(np.float32)
    prof = np.median(lum, axis=0)
    steps = np.diff(prof)

    if p.trim == "auto":
        t = 0
        while t < min(TRIM_CAP, len(steps)) and abs(steps[t]) > BLOOM_STEP:
            t += 1
    else:
        try:
            t = max(0, min(int(p.trim), W // 4))
        except ValueError:
            raise FileError(f"--trim must be 'auto' or an integer, "
                            f"got {p.trim!r}") from None

    K = min(p.sample, W - t)
    # keep the band outside the border's inner boundary (frame line/artwork).
    # its signature: a bright positive jump after the border's darkening ramp,
    # or any step far above the local gradient — a merely strong bloom
    # gradient must NOT count.
    for i in range(t + 2, len(steps)):
        local = float(np.median(np.abs(steps[t:i]))) if i > t else 0.0
        if steps[i] > INNER_STEP or abs(steps[i]) > max(INNER_STEP,
                                                        3 * local + 3):
            K = min(K, i - t)
            break
    if K < 1:
        raise FileError(f"no usable border band (width {W}px, trim {t})")
    return t, K, prof


def _source_rows(H: int, n_total: int, ramp: np.ndarray, p: Params,
                 rng: np.random.Generator) -> np.ndarray:
    """Along-edge source coordinate for each output pixel.

    Two smoothed displacement fields are combined: a small local wobble
    (kills repeated-fleck trails along the extension) and a long-range
    patch shuffle (borrows texture from elsewhere on the edge so patterns
    do not near-repeat across mirror passes). Both ramp in from the seam.
    """
    rows = np.broadcast_to(np.arange(H, dtype=np.intp)[:, None],
                           (H, n_total)).astype(np.float32)
    disp = np.zeros((H, n_total), dtype=np.float32)
    if p.jitter_cross > 0:
        disp += (smooth_field(rng, (H, n_total), p.jitter_smooth,
                              standardize=True) * p.jitter_cross)
    if p.shuffle > 0:
        disp += (smooth_field(rng, (H, n_total), SHUFFLE_SMOOTH,
                              standardize=True) * p.shuffle)
    rows += disp * ramp[None, :]
    out = np.rint(rows).astype(np.intp)
    np.clip(out, 0, H - 1, out=out)
    return out


def _guard_rows(rows: np.ndarray, H: int, G: int) -> np.ndarray:
    """Clamp source rows near band ends so rounded/white scan corners never
    seed the extension."""
    rows[:G] = np.clip(rows[:G], G, H - 1 - G)
    rows[H - G:] = np.clip(rows[H - G:], G, H - 1 - G)
    return rows


def synth_left(img: np.ndarray, n: int, p: Params, rng: np.random.Generator,
               overwrite: bool, notes: list[str],
               geom: tuple[int, int] | None = None) -> tuple[np.ndarray, int]:
    """Synthesize the extension for the LEFT edge of img (H,W,C float32).

    Returns (ext, t_eff): ext has shape (H, n_total, C) with columns ordered
    from the seam outward; t_eff is how many original columns the caller must
    drop (halo overwrite) — 0 in blend mode.
    """
    H, W, C = img.shape
    t, K = geom if geom is not None else edge_geometry(img, p)[:2]
    if K < p.sample:
        notes.append(f"band clamped to {K}px (inner border structure detected)")
    if p.trim == "auto" and t:
        notes.append(f"auto-trimmed {t}px of scanner bloom")

    band = img[:, t:t + K].astype(np.float32)  # depth 0 = outermost clean line
    n_total = n + (t if overwrite else 0)

    j = np.arange(1, n_total + 1, dtype=np.float32)                # 1 = at seam
    ramp = np.clip((j - 1) / max(p.seam_feather, 1e-6), 0.0, 1.0)  # (n_total,)
    period = max(2 * K - 2, 1)
    tri = (K - 1) - np.abs((K - 1) - ((j.astype(np.intp) - 1) % period))

    G = p.corner_guard
    guard = p.mode == "smart" and G > 0 and H > 2 * G

    if p.mode == "naive" or K == 1:
        ext = np.repeat(band[:, :1], n_total, axis=1)
    else:
        # tone/texture split: the border tone (smooth along the edge, often a
        # real gradient across it) is continued outward MIRRORED — ordered,
        # seam-continuous, never extrapolated — while the speckle residual is
        # resampled with jittered depth + along-edge displacement.
        tone = conv_axis(band, gauss_kernel(TONE_SIGMA), 0)        # (H, K, C)
        resid = band - tone

        tone_rows = np.arange(H, dtype=np.intp)
        if guard:
            tone_rows = np.clip(tone_rows, G, H - 1 - G)
        ext_tone = tone[tone_rows][:, tri]                         # (H, n_total, C)

        U = smooth_field(rng, (H, n_total), p.jitter_smooth)
        jit = np.float32(p.jitter) * ramp
        depth = np.rint((1 - jit)[None, :] * tri[None, :]
                        + jit[None, :] * U * (K - 1)).astype(np.intp)
        depth = np.clip(depth, 0, K - 1)

        rows = _source_rows(H, n_total, ramp, p, rng)
        if guard:
            rows = _guard_rows(rows, H, G)

        ext = ext_tone + resid[rows, depth]                        # (H, n_total, C)

    # halo blend (JPEG): anchor the seam on the true outermost line, then
    # dissolve it into clean band texture over the feather ramp
    if not overwrite and t > 0:
        w = ramp[None, :, None]
        ext = w * ext + (1 - w) * img[:, 0].astype(np.float32)[:, None, :]

    # noise matched to the band's own grain (color channels only)
    if p.noise > 0:
        nch = C - 1 if C in (2, 4) else C
        sigma = np.clip(highpass_std(band[..., :nch]), 0.0, 12.0)
        ext[..., :nch] += (rng.standard_normal((H, n_total, nch)).astype(np.float32)
                           * (p.noise * sigma)[None, None, :] * ramp[None, :, None])

    # smudge: isotropic blur ramped toward the outer cut edge, synth only
    if p.smudge > 0 and n_total > 1:
        w = ((j - 1) / n_total)[None, :, None]
        ext = ext * (1 - w) + gaussian_blur2d(ext, p.smudge) * w

    return ext, (t if overwrite else 0)


def extend_edge(img: np.ndarray, side: str, n: int, p: Params,
                rng: np.random.Generator, overwrite: bool, notes: list[str],
                geom: tuple[int, int] | None = None) -> np.ndarray:
    k = _SIDE_ROT[side]
    a = np.ascontiguousarray(np.rot90(img, k))
    ext, t_eff = synth_left(a, n, p, rng, overwrite, notes, geom)
    out = np.concatenate([ext[:, ::-1], a[:, t_eff:]], axis=1)
    return np.ascontiguousarray(np.rot90(out, -k))


def extend_image(arr: np.ndarray, extents: tuple[int, int, int, int], p: Params,
                 rng: np.random.Generator, overwrite: bool,
                 notes: list[str]) -> np.ndarray:
    """Two-pass extension: sides first, then top/bottom on the widened image
    (corners inherit freshly synthesized side texture). Edge analysis always
    runs on the ORIGINAL image so first-pass synthesis can't skew it."""
    eL, eT, eR, eB = extents
    out = arr.astype(np.float32)
    passes = (("left", eL), ("right", eR), ("top", eT), ("bottom", eB))
    geoms = {side: edge_geometry(
                 np.ascontiguousarray(np.rot90(out, _SIDE_ROT[side])), p)[:2]
             for side, n in passes if n > 0}
    for side, n in passes:
        if n > 0:
            out = extend_edge(out, side, n, p, rng, overwrite, notes,
                              geoms[side])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
