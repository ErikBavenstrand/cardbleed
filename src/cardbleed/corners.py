"""Corner squaring: fill edge-connected background with the nearest border.

Used before bleeding so a rounded-corner or ragged scan becomes a clean
square-cornered rectangle. Only background pixels (transparent, near-black, or
near-white, reachable from the image edge) are touched; opaque artwork never is.
"""

from __future__ import annotations

import numpy as np

# background = anything that isn't the card border — transparent, or an extreme
# (near-black / near-white) scan margin — reachable from the image edge.
_ALPHA = 128.0
_DARK, _BRIGHT = 45.0, 232.0
_MAX_FRAC = 0.15  # more edge background than this = card floats in a margin, skip


def square_background(arr: np.ndarray, notes: list[str]) -> np.ndarray:
    """Fill edge-connected background with the nearest border pixel.

    A no-op when there's no such background, and skipped when the card floats in
    a large margin (not a corner case).
    """
    h, w = arr.shape[:2]
    c = arr.shape[2] if arr.ndim == 3 else 1
    lum = arr[..., :3].mean(axis=2) if c >= 3 else arr[..., 0].astype(np.float32)
    bg = (lum < _DARK) | (lum > _BRIGHT)
    if c in (2, 4):
        bg = bg | (arr[..., -1] < _ALPHA)
    if not bg.any():
        return arr
    ext = np.zeros((h, w), dtype=bool)
    ext[0], ext[-1], ext[:, 0], ext[:, -1] = bg[0], bg[-1], bg[:, 0], bg[:, -1]
    while True:  # flood inward through contiguous background from the edges
        g = ext.copy()
        g[1:] |= ext[:-1] & bg[1:]
        g[:-1] |= ext[1:] & bg[:-1]
        g[:, 1:] |= ext[:, :-1] & bg[:, 1:]
        g[:, :-1] |= ext[:, 1:] & bg[:, :-1]
        if np.array_equal(g, ext):
            break
        ext = g
    if not ext.any() or ext.mean() > _MAX_FRAC:
        return arr
    out = arr.astype(np.float32)
    known = ~ext
    for _ in range(h + w):  # grow the border into the background, nearest-first
        if known.all():
            break
        need = ext & ~known
        if not need.any():
            break
        got = np.zeros((h, w), dtype=bool)
        cand = out.copy()
        for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            rk, ro = np.roll(known, sh, ax), np.roll(out, sh, ax)
            if ax == 0:  # kill np.roll wrap-around at the edge it came from
                rk[0 if sh > 0 else -1] = False
            else:
                rk[:, 0 if sh > 0 else -1] = False
            take = need & ~got & rk
            if take.any():
                cand[take] = ro[take]
                got |= take
        out[got] = cand[got]
        known |= got
    if c in (2, 4):
        out[..., -1] = np.where(ext, 255.0, out[..., -1])
    notes.append("fill-corners: squared background into the border")
    return np.clip(np.rint(out), 0, 255).astype(arr.dtype)
