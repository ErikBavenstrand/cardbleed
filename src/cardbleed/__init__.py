"""Extend the borders of card scans outward for printing.

Continues the existing border pattern (holo speckle, solid colors, ...)
uniformly on all four edges without ever degrading the original image data:

  PNG  -> PNG   original pixels bit-identical (lossless re-serialize)
  WebP -> WebP  written lossless; decoded original pixels preserved exactly
  JPEG -> JPEG  DCT-domain surgery: original coefficient blocks are copied
                bit-exact into a larger grid; only new border blocks are
                encoded (with the original's own quantization tables)

Examples:
  cardbleed card.png --compare
  cardbleed ./cards/ -e 20 --recursive
  cardbleed card.jpg -e 2.5mm --fix-aspect
  cardbleed card.png --target 69x94mm
  cardbleed --selfcheck card.png

Modes:
  --mode smart (default)  resample a K-px band just inside the edge with a
                          smoothed random depth per pixel: speckle re-randomizes
                          instead of streaking, flat borders stay flat.
  --mode naive            replicate the outermost clean line straight outward
                          (plus the same configurable noise + smudge).
"""

from __future__ import annotations

import math
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rich_click as click
from PIL import Image
from rich.console import Console
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn)

__version__ = "0.1.0"

FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}
MAGENTA = np.array([255, 0, 255], dtype=np.uint8)


class FileError(Exception):
    """Per-file error: reported, file skipped, batch continues."""


@dataclass
class Params:
    mode: str = "smart"
    sample: int = 8
    trim: str = "auto"          # "auto" or an integer as string
    jitter: float = 0.85
    jitter_smooth: float = 1.2
    jitter_cross: float = 4.0
    noise: float = 0.35
    smudge: float = 0.6
    seam_feather: int = 3
    corner_guard: int = 12
    halo: str = "auto"  # auto | overwrite | blend

TRIM_CAP = 3          # max lines auto-trim may cut
BLOOM_STEP = 12.0     # luminance step marking a scanner bloom line
INNER_STEP = 15.0     # luminance step marking the border's inner boundary
TONE_SIGMA = 6.0      # along-edge smoothing that separates tone from speckle


# --------------------------------------------------------------------------
# small numpy helpers (no SciPy)
# --------------------------------------------------------------------------

def _conv_axis(a: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
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


def _gauss_kernel(sigma: float) -> np.ndarray:
    r = max(1, math.ceil(3 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def gaussian_blur2d(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur over the first two axes of (H,W) or (H,W,C)."""
    if sigma <= 0:
        return a.astype(np.float32, copy=True)
    k = _gauss_kernel(sigma)
    return _conv_axis(_conv_axis(a, k, 0), k, 1)


def box_blur3(a: np.ndarray) -> np.ndarray:
    k = np.full(3, 1 / 3, dtype=np.float32)
    return _conv_axis(_conv_axis(a, k, 0), k, 1)


def highpass_std(region: np.ndarray) -> np.ndarray:
    """Per-channel std of the high-frequency residual of an (H,W,C) region."""
    return (region.astype(np.float32) - box_blur3(region)).std(axis=(0, 1))


# --------------------------------------------------------------------------
# core synthesis
# --------------------------------------------------------------------------

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

    j = np.arange(1, n_total + 1, dtype=np.float32)               # 1 = at seam
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
        # resampled with jittered depth + along-edge wobble.
        tone = _conv_axis(band, _gauss_kernel(TONE_SIGMA), 0)     # (H, K, C)
        resid = band - tone

        tone_rows = np.arange(H, dtype=np.intp)
        if guard:
            tone_rows = np.clip(tone_rows, G, H - 1 - G)
        ext_tone = tone[tone_rows][:, tri]                        # (H, n_total, C)

        U = gaussian_blur2d(rng.random((H, n_total), dtype=np.float32),
                            p.jitter_smooth)
        U = (U - U.min()) / max(np.ptp(U), 1e-6)
        jit = np.float32(p.jitter) * ramp
        depth = np.rint((1 - jit)[None, :] * tri[None, :]
                        + jit[None, :] * U * (K - 1)).astype(np.intp)
        depth = np.clip(depth, 0, K - 1)

        rows = np.broadcast_to(np.arange(H, dtype=np.intp)[:, None],
                               (H, n_total)).copy()
        if p.jitter_cross > 0:
            V = gaussian_blur2d(rng.random((H, n_total), dtype=np.float32),
                                p.jitter_smooth)
            V = (V - V.mean()) / max(float(V.std()), 1e-6)
            rows += np.rint(V * p.jitter_cross * ramp[None, :]).astype(np.intp)
            np.clip(rows, 0, H - 1, out=rows)
        if guard:
            rows[:G] = np.clip(rows[:G], G, H - 1 - G)
            rows[H - G:] = np.clip(rows[H - G:], G, H - 1 - G)

        ext = ext_tone + resid[rows, depth]                       # (H, n_total, C)

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


_SIDE_ROT = {"left": 0, "top": 1, "right": 2, "bottom": 3}  # CCW turns to face left


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


# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------

def parse_amount(s: str) -> tuple[float, str]:
    raw, s = s, s.strip().lower()
    unit = "px"
    for u in ("mm", "px"):
        if s.endswith(u):
            unit, s = u, s[:-len(u)]
    try:
        return float(s), unit
    except ValueError:
        raise FileError(f"cannot parse amount {raw!r} (expected e.g. "
                        "'16', '16px' or '2.5mm')") from None


def parse_size(s: str) -> tuple[float, float, str]:
    raw, s = s, s.strip().lower()
    unit = "px"
    for u in ("mm", "px"):
        if s.endswith(u):
            unit, s = u, s[:-len(u)]
    w, sep, h = s.partition("x")
    try:
        if not sep:
            raise ValueError
        return float(w), float(h), unit
    except ValueError:
        raise FileError(f"cannot parse size {raw!r} (expected WxH, e.g. "
                        "'69x94mm' or '440x600')") from None


def resolve_extents(args, size: tuple[int, int],
                    mcu: tuple[int, int] | None,
                    notes: list[str]) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) extension in px.

    Order of composition: optional aspect-ratio fill first, then the uniform /
    per-edge / target extension on top. For JPEG (mcu set), left/top must be
    MCU-aligned; the remainder is shifted to right/bottom so final dimensions
    (and thus aspect ratio and --target sizes) stay exact.
    """
    W0, H0 = size
    card_w, card_h, _ = parse_size(args.card_size)   # unit ignored: always mm
    ppm_x, ppm_y = W0 / card_w, H0 / card_h

    # -- aspect verification / fill ----------------------------------------
    ratio = card_w / card_h
    want_h = round(W0 / ratio)
    want_w = round(H0 * ratio)
    aL = aT = aR = aB = 0
    off = not (want_h == H0 or want_w == W0)
    if args.fix_aspect:
        if want_h > H0:      # too wide -> grow height
            pad = want_h - H0
            aT, aB = pad // 2, pad - pad // 2
            ppm_y = ppm_x    # unpadded axis is the physical reference now
        elif want_w > W0:    # too tall -> grow width
            pad = want_w - W0
            aL, aR = pad // 2, pad - pad // 2
            ppm_x = ppm_y
        if aL or aT or aR or aB:
            notes.append(f"aspect fill: +{aL}/+{aR} left/right, +{aT}/+{aB} "
                         f"top/bottom to reach {card_w:g}:{card_h:g}")
    elif off:
        notes.append(f"aspect ratio {W0}x{H0} deviates from "
                     f"{card_w:g}:{card_h:g} (want ~{want_w}x{H0} or "
                     f"{W0}x{want_h}); use --fix-aspect to pad it square")

    def to_px(amount: str, ppm: float) -> int:
        v, unit = parse_amount(amount)
        px = int(round(v * ppm)) if unit == "mm" else int(round(v))
        if px < 0:
            raise FileError(f"negative extension {amount!r}")
        return px

    # -- main extension ------------------------------------------------------
    if args.target:
        if any(v is not None for v in (args.left, args.right, args.top,
                                       args.bottom)):
            raise FileError("--target cannot be combined with per-edge "
                            "overrides (--left/--right/--top/--bottom)")
        tw, th, unit = parse_size(args.target)
        tw_px = int(round(tw * ppm_x)) if unit == "mm" else int(round(tw))
        th_px = int(round(th * ppm_y)) if unit == "mm" else int(round(th))
        aw, ah = W0 + aL + aR, H0 + aT + aB
        if tw_px < aw or th_px < ah:
            raise FileError(f"--target {args.target} is smaller than "
                            f"{aw}x{ah} (image incl. aspect fill)")
        eL = (tw_px - aw) // 2
        eR = tw_px - aw - eL
        eT = (th_px - ah) // 2
        eB = th_px - ah - eT
    else:
        eL = to_px(args.left if args.left is not None else args.extend, ppm_x)
        eR = to_px(args.right if args.right is not None else args.extend, ppm_x)
        eT = to_px(args.top if args.top is not None else args.extend, ppm_y)
        eB = to_px(args.bottom if args.bottom is not None else args.extend, ppm_y)

    L, T, R, B = aL + eL, aT + eT, aR + eR, aB + eB

    # -- JPEG MCU alignment: shift remainders left->right / top->bottom -----
    if mcu is not None:
        mw, mh = mcu
        if L % mw:
            shift = L % mw
            L, R = L - shift, R + shift
            notes.append(f"JPEG alignment: moved {shift}px from left to right "
                         f"edge (left offset must be a multiple of {mw})")
        if T % mh:
            shift = T % mh
            T, B = T - shift, B + shift
            notes.append(f"JPEG alignment: moved {shift}px from top to bottom "
                         f"edge (top offset must be a multiple of {mh})")
    return L, T, R, B


# --------------------------------------------------------------------------
# image I/O
# --------------------------------------------------------------------------

def load_pixels(path: Path, fmt: str, notes: list[str]):
    """Decode to (H,W,C) uint8 + a dict of metadata for the writer."""
    if fmt == "jpeg":
        import jpeglib
        dct = jpeglib.read_dct(str(path))
        cs = str(getattr(dct, "jpeg_color_space", ""))
        if dct.num_components == 3 and "YCbCr" not in cs:
            raise FileError(f"JPEG color space {cs or 'unknown'} not supported "
                            "(only YCbCr and grayscale)")
        spat = jpeglib.read_spatial(str(path))
        arr = np.asarray(spat.spatial)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.shape[2] not in (1, 3):
            raise FileError(f"unsupported JPEG color layout ({arr.shape[2]} "
                            "channels; CMYK/YCCK not supported)")
        return arr, {"dct": dct}

    im = Image.open(path)
    if fmt == "webp" and getattr(im, "is_animated", False):
        raise FileError("animated WebP not supported")
    meta = {"icc_profile": im.info.get("icc_profile"),
            "exif": im.info.get("exif")}
    if im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        notes.append(f"palette image promoted to {im.mode} (pixel values identical)")
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        raise FileError(f"{im.mode} (high bit depth) not supported; "
                        "convert to 8-bit first")
    if im.mode not in ("L", "LA", "RGB", "RGBA"):
        im = im.convert("RGB")
        notes.append("converted to RGB")
    meta["mode"] = im.mode
    arr = np.asarray(im)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr, meta


def save_png_webp(arr: np.ndarray, meta: dict, out: Path, fmt: str,
                  dpi: tuple[float, float]) -> None:
    mode = meta.get("mode", "RGB")
    im = Image.fromarray(arr[:, :, 0] if arr.shape[2] == 1 else arr, mode=mode)
    kw = {}
    if meta.get("icc_profile"):
        kw["icc_profile"] = meta["icc_profile"]
    if meta.get("exif"):
        kw["exif"] = meta["exif"]
    if fmt == "png":
        im.save(out, format="PNG", dpi=dpi, **kw)
    else:
        # exact=True: keep RGB values under fully-transparent alpha
        im.save(out, format="WEBP", lossless=True, quality=80, exact=True, **kw)


# --------------------------------------------------------------------------
# JPEG DCT surgery
# --------------------------------------------------------------------------

_DCT_M = None


def _dct_matrix() -> np.ndarray:
    global _DCT_M
    if _DCT_M is None:
        i = np.arange(8, dtype=np.float64)[:, None]
        j = np.arange(8, dtype=np.float64)[None, :]
        m = np.cos((2 * j + 1) * i * np.pi / 16)
        m[0] *= math.sqrt(1 / 8)
        m[1:] *= math.sqrt(2 / 8)
        _DCT_M = m
    return _DCT_M


def _plane_to_qblocks(plane: np.ndarray, qt: np.ndarray) -> np.ndarray:
    """(h,w) float plane -> quantized DCT blocks (bv,bh,8,8) int16."""
    h, w = plane.shape
    bv, bh = -(-h // 8), -(-w // 8)
    plane = np.pad(plane, ((0, bv * 8 - h), (0, bh * 8 - w)), mode="edge")
    blocks = plane.reshape(bv, 8, bh, 8).transpose(0, 2, 1, 3) - 128.0
    D = _dct_matrix()
    coeff = np.einsum("ik,vhkl,jl->vhij", D, blocks, D)
    return np.rint(coeff / qt[None, None]).astype(np.int16)


def _subsample(plane: np.ndarray, fx: int, fy: int) -> np.ndarray:
    if fx == 1 and fy == 1:
        return plane
    h, w = plane.shape
    plane = np.pad(plane, ((0, (-h) % fy), (0, (-w) % fx)), mode="edge")
    return plane.reshape(plane.shape[0] // fy, fy,
                         plane.shape[1] // fx, fx).mean(axis=(1, 3))


def jpeg_factors(dct, ci: int) -> tuple[int, int]:
    """Subsampling divisors (fx, fy) of component ci.

    jpeglib's samp_factor rows are ordered [vertical, horizontal].
    """
    sf = np.asarray(dct.samp_factor)
    hmax, vmax = int(sf[:, 1].max()), int(sf[:, 0].max())
    return hmax // int(sf[ci, 1]), vmax // int(sf[ci, 0])


def jpeg_mcu(dct) -> tuple[int, int]:
    """(mcu_width, mcu_height) in luma pixels."""
    sf = np.asarray(dct.samp_factor)
    return 8 * int(sf[:, 1].max()), 8 * int(sf[:, 0].max())


def jpeg_paste_box(dct, ci: int, size0: tuple[int, int],
                   extents: tuple[int, int, int, int],
                   halo_overwrite: bool, trim_px: int = TRIM_CAP,
                   ) -> tuple[slice, slice, int, int]:
    """Block region of component ci that stays bit-exact.

    Returns (rows, cols, off_v, off_h): slices into the NEW block grid and the
    placement offset of the original component blocks.
    """
    W0, H0 = size0
    L, T, R, B = extents
    fx, fy = jpeg_factors(dct, ci)
    comps = [dct.Y, dct.Cb, dct.Cr][ci] if dct.Cb is not None else dct.Y
    ov, oh = comps.shape[:2]
    off_v, off_h = (T // fy) // 8, (L // fx) // 8
    ch0, cw0 = -(-H0 // fy), -(-W0 // fx)      # component pixel dims (original)
    lo_v, lo_h = off_v, off_h
    hi_v, hi_h = off_v + ov, off_h + oh
    # straddling last block row/col contains encoder padding that would become
    # visible next to a new extension -> re-encode it (content preserved in
    # pixel space; requantized with the original tables)
    if B > 0 and ch0 % 8:
        hi_v -= 1
    if R > 0 and cw0 % 8:
        hi_h -= 1
    if halo_overwrite:
        # opt-in: re-encode the outer block ring so the rewritten halo pixels
        # actually land in the file — only on edges that were extended, and
        # deep enough to cover the trimmed pixels
        rx = max(1, -(-trim_px // (8 * fx)))
        ry = max(1, -(-trim_px // (8 * fy)))
        if L > 0:
            lo_h += rx
        if T > 0:
            lo_v += ry
        if R > 0:
            hi_h -= rx
        if B > 0:
            hi_v -= ry
    return slice(lo_v, max(lo_v, hi_v)), slice(lo_h, max(lo_h, hi_h)), off_v, off_h


def save_jpeg(arr: np.ndarray, meta: dict, out: Path, size0: tuple[int, int],
              extents: tuple[int, int, int, int], halo_overwrite: bool,
              trim_px: int = TRIM_CAP) -> None:
    import jpeglib
    dct = meta["dct"]
    H1, W1, C = arr.shape
    ncomp = dct.num_components
    f = arr.astype(np.float32)
    if ncomp == 1:
        planes = [f[:, :, 0]]
    else:
        R, G, B = f[:, :, 0], f[:, :, 1], f[:, :, 2]
        planes = [0.299 * R + 0.587 * G + 0.114 * B,
                  128.0 - 0.168735892 * R - 0.331264108 * G + 0.5 * B,
                  128.0 + 0.5 * R - 0.418687589 * G - 0.081312411 * B]

    orig = [dct.Y, dct.Cb, dct.Cr][:ncomp]
    comps = []
    for ci in range(ncomp):
        fx, fy = jpeg_factors(dct, ci)
        qt = dct.get_component_qt(ci)
        plane = _subsample(planes[ci], fx, fy)
        blocks = _plane_to_qblocks(plane, np.asarray(qt, dtype=np.float64))
        rows, cols, off_v, off_h = jpeg_paste_box(dct, ci, size0, extents,
                                                  halo_overwrite, trim_px)
        src = orig[ci]
        blocks[rows, cols] = src[rows.start - off_v: rows.stop - off_v,
                                 cols.start - off_h: cols.stop - off_h]
        comps.append(np.ascontiguousarray(blocks))

    new = jpeglib.from_dct(Y=comps[0],
                           Cb=comps[1] if ncomp > 1 else None,
                           Cr=comps[2] if ncomp > 1 else None,
                           qt=dct.qt,
                           quant_tbl_no=list(np.asarray(dct.quant_tbl_no)))
    new.width, new.height = W1, H1
    try:
        new.markers = dct.markers
    except Exception:
        pass
    new.write_dct(str(out))


# --------------------------------------------------------------------------
# batch driver
# --------------------------------------------------------------------------

def iter_inputs(paths: list[str], recursive: bool,
                suffix: str) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    errors: list[str] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            for f in sorted(it):
                if f.suffix.lower() not in FORMATS or not f.is_file():
                    continue
                if suffix and f.stem.endswith(suffix):
                    continue  # already an output of this tool
                if f.stem.endswith("_compare"):
                    continue
                files.append(f)
        elif p.is_file():
            files.append(p)
        else:
            errors.append(f"{p}: no such file or directory")
    return files, errors


def make_compare(orig: np.ndarray, result: np.ndarray,
                 extents: tuple[int, int, int, int]) -> np.ndarray:
    def rgb(a):
        if a.shape[2] < 3:  # L / LA: show the gray channel
            return np.repeat(a[:, :, :1], 3, axis=2)
        return a[:, :, :3]

    o, r = rgb(orig), rgb(result)
    marked = r.copy()
    L, T, _, _ = extents
    h0, w0 = o.shape[:2]
    marked[T, L:L + w0] = MAGENTA
    marked[T + h0 - 1, L:L + w0] = MAGENTA
    marked[T:T + h0, L] = MAGENTA
    marked[T:T + h0, L + w0 - 1] = MAGENTA

    gut, pad = 12, 12
    hmax = max(o.shape[0], r.shape[0])
    wsum = o.shape[1] + r.shape[1] + marked.shape[1] + 2 * gut + 2 * pad
    sheet = np.full((hmax + 2 * pad, wsum, 3), 96, dtype=np.uint8)
    x = pad
    for panel in (o, r, marked):
        sheet[pad:pad + panel.shape[0], x:x + panel.shape[1]] = panel
        x += panel.shape[1] + gut
    return sheet


def process_file(path: Path, args, console: Console | None = None,
                 claimed: dict | None = None) -> Path:
    fmt = FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise FileError(f"unsupported format {path.suffix!r} "
                        "(png/jpg/jpeg/webp only)")
    notes: list[str] = []
    arr, meta = load_pixels(path, fmt, notes)
    H0, W0 = arr.shape[:2]

    mcu = jpeg_mcu(meta["dct"]) if fmt == "jpeg" else None
    extents = resolve_extents(args, (W0, H0), mcu, notes)
    eL, eT, eR, eB = extents

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else path.parent
    out = out_dir / f"{path.stem}{args.suffix}{path.suffix.lower()}"
    cmp_path = out_dir / f"{path.stem}{args.suffix}_compare.png"
    for target in (out, cmp_path) if args.compare else (out,):
        if target.exists() and target.resolve() == path.resolve():
            raise FileError(f"output path {target.name} equals the input; "
                            "refusing to overwrite the source "
                            "(set --suffix or --out-dir)")
    if claimed is not None:
        key = str(out.resolve()) if out.is_absolute() else str(out)
        if key in claimed:
            raise FileError(f"output {out.name} was already produced by "
                            f"{claimed[key]} in this batch (same stem); "
                            "use --out-dir per folder or rename inputs")
        claimed[key] = path.name
    if not any(extents):
        raise FileError("nothing to do (all extension amounts are 0)")

    if args.dry_run:
        if console is not None:
            clash = (" [red](exists: needs --force)[/]"
                     if out.exists() and not args.force else "")
            console.print(
                f"[bold cyan]{path.name}[/] {W0}×{H0} → "
                f"[bold]{W0+eL+eR}×{H0+eT+eB}[/] "
                f"[dim](+L{eL} +T{eT} +R{eR} +B{eB})[/] → {out}{clash}")
            for n in dict.fromkeys(notes):
                console.print(f"   [dim]• {n}[/]")
        return out
    if out.exists() and not args.force:
        raise FileError(f"{out.name} exists (use --force to overwrite outputs)")

    p = Params(mode=args.mode, sample=args.sample, trim=args.trim,
               jitter=args.jitter, jitter_smooth=args.jitter_smooth,
               jitter_cross=args.jitter_cross, noise=args.noise,
               smudge=args.smudge, seam_feather=args.seam_feather,
               corner_guard=args.corner_guard, halo=args.halo)
    halo = p.halo if p.halo != "auto" else ("blend" if fmt == "jpeg" else "overwrite")
    overwrite = halo == "overwrite"
    if fmt == "jpeg" and overwrite:
        notes.append("halo overwrite on JPEG: the outer block ring is "
                     "re-encoded (localized loss inside the border)")

    rng = np.random.default_rng([args.seed, zlib.crc32(path.name.encode())])
    result = extend_image(arr, extents, p, rng, overwrite, notes)

    out_dir.mkdir(parents=True, exist_ok=True)
    card_w, card_h, _ = parse_size(args.card_size)
    dpi = (W0 / card_w * 25.4, H0 / card_h * 25.4)
    if args.fix_aspect:
        ref = dpi[0] if W0 / H0 > card_w / card_h else dpi[1]
        dpi = (ref, ref)
    if fmt == "jpeg":
        trim_px = TRIM_CAP if args.trim == "auto" else int(args.trim)
        save_jpeg(result, meta, out, (W0, H0), extents, overwrite, trim_px)
    else:
        save_png_webp(result, meta, out, fmt, dpi)

    if args.compare:
        sheet = make_compare(arr, result, extents)
        Image.fromarray(sheet).save(cmp_path)
        notes.append(f"comparison sheet: {cmp_path.name}")

    if console is not None:
        console.print(
            f"[bold cyan]{path.name}[/] {W0}×{H0} → "
            f"[bold]{result.shape[1]}×{result.shape[0]}[/] "
            f"[dim](+L{eL} +T{eT} +R{eR} +B{eB} · {args.mode} · halo={halo})[/]"
            f" → [green]{out.name}[/]")
        for n in dict.fromkeys(notes):  # dedupe, keep order
            console.print(f"   [dim]• {n}[/]")
    return out


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def _mk_args(base, **over):
    ns = SimpleNamespace(**vars(base))
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def selfcheck(args) -> int:
    console = Console(highlight=False)
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = ""):
        results.append((name, bool(ok), detail))
        mark = "[green]✓ PASS[/]" if ok else "[red]✗ FAIL[/]"
        console.print(f"  {mark}  {name}" + (f"  [dim]({detail})[/]" if detail else ""))

    tmp = Path(tempfile.mkdtemp(prefix="extend_border_check_"))
    console.print(f"[dim]selfcheck workspace: {tmp}[/]")
    base = _mk_args(args, out_dir=str(tmp / "out"), force=True, compare=False,
                    dry_run=False, suffix="_ext")

    # fixtures ---------------------------------------------------------------
    def bordered(w, h, border, inner, bw=24, noise_sigma=0.0):
        a = np.full((h, w, 3), inner, dtype=np.float64)
        m = np.zeros((h, w), dtype=bool)
        m[:bw] = m[-bw:] = True
        m[:, :bw] = m[:, -bw:] = True
        a[m] = border
        if noise_sigma:
            rng = np.random.default_rng(7)
            a[m] += rng.normal(0, noise_sigma, (int(m.sum()), 3))
        return np.clip(a, 0, 255).astype(np.uint8)

    flat_p = tmp / "flat.png"
    Image.fromarray(bordered(300, 420, (255, 214, 0), (255, 255, 255))).save(flat_p)
    speck_p = tmp / "speckle.png"
    Image.fromarray(bordered(300, 420, (180, 150, 90), (120, 120, 120),
                             noise_sigma=25)).save(speck_p)
    grad_p = tmp / "grad.png"
    g = np.linspace(30, 220, 48)[:, None, None] * np.ones((48, 64, 3))
    Image.fromarray(g.astype(np.uint8)).save(grad_p)

    real_p = None
    for cand in args.inputs or []:
        c = Path(cand).expanduser()
        if c.is_file() and c.suffix.lower() == ".png":
            real_p = c
            break

    # A: geometry, px
    o = process_file(flat_p, _mk_args(base, extend="16"))
    check("geometry px (+16 all edges)", Image.open(o).size == (332, 452),
          f"{Image.open(o).size}")

    # B: orientation — extend top only on an asymmetric gradient
    o = process_file(grad_p, _mk_args(base, top="8", left="0", right="0",
                                      bottom="0", trim=0))
    ga = np.asarray(Image.open(o))
    check("orientation (top-only)", ga.shape[:2] == (56, 64)
          and np.array_equal(ga[8:], np.asarray(Image.open(grad_p)))
          and 20 < ga[2].mean() < 90,
          f"shape {ga.shape[:2]}, top mean {ga[2].mean():.0f}")

    # C: interior identity, PNG (outside the auto-trim cap ring)
    if real_p is not None:
        src = np.asarray(Image.open(real_p).convert("RGB"))
        o = process_file(real_p, _mk_args(base, extend="16"))
        outa = np.asarray(Image.open(o))
        t = TRIM_CAP
        crop = outa[16 + t:16 + src.shape[0] - t, 16 + t:16 + src.shape[1] - t]
        check("interior identity PNG", np.array_equal(crop, src[t:-t, t:-t]))

        # D: determinism
        o2 = process_file(real_p, _mk_args(base, extend="16"))
        check("determinism (same seed)", o.read_bytes() == o2.read_bytes())

        # E: geometry, mm
        o = process_file(real_p, _mk_args(base, extend="2mm"))
        W0, H0 = src.shape[1], src.shape[0]
        exp = (W0 + 2 * round(2 * W0 / 63), H0 + 2 * round(2 * H0 / 88))
        check("geometry mm", Image.open(o).size == exp,
              f"{Image.open(o).size} vs {exp}")

        # F: geometry, target
        o = process_file(real_p, _mk_args(base, target="69x94mm"))
        exp = (round(69 * W0 / 63), round(94 * H0 / 88))
        check("geometry target", Image.open(o).size == exp,
              f"{Image.open(o).size} vs {exp}")

        # G: DPI stamp
        got = Image.open(o).info.get("dpi", (0, 0))
        exp_dpi = (W0 / 63 * 25.4, H0 / 88 * 25.4)
        check("DPI stamp", abs(got[0] - exp_dpi[0]) < 1.5 and
              abs(got[1] - exp_dpi[1]) < 1.5, f"{got}")

        # H: texture statistics + streaks + seam on a real textured scan
        # (fixed trim=1 so band positions are known)
        o = process_file(real_p, _mk_args(base, extend="16", trim="1",
                                          suffix="_ext_t1"))
        outa = np.asarray(Image.open(o)).astype(np.float32)
        t, K = 1, 7
        synth = outa[100:-100, :16]          # left extension, away from corners
        band = outa[100:-100, 16 + t:16 + t + K]
        rs, rb = highpass_std(synth).mean(), highpass_std(band).mean()
        check("texture stats (residual std ratio)", 0.5 <= rs / max(rb, 1e-6) <= 1.6,
              f"ratio {rs / max(rb, 1e-6):.2f}")
        # the mirrored tone keeps the extension's mean near the band mean
        mean_d = abs(synth.mean(axis=(0, 1)) - band.mean(axis=(0, 1))).max()
        check("texture stats (mean vs band within 8/255)", mean_d <= 8.0,
              f"{mean_d:.1f}")

        hp = synth - box_blur3(synth)
        def lag3(x):
            a, b = x[:, :-3].ravel(), x[:, 3:].ravel()
            a = a - a.mean(); b = b - b.mean()
            den = math.sqrt(float((a * a).sum() * (b * b).sum()))
            return float((a * b).sum()) / max(den, 1e-6)
        hb = band - box_blur3(band)
        check("no streaks (lag-3 autocorr)", lag3(hp.mean(axis=2)) <=
              max(1.5 * abs(lag3(hb.mean(axis=2))), 0.35),
              f"synth {lag3(hp.mean(axis=2)):.2f} vs band {lag3(hb.mean(axis=2)):.2f}")

        seam_x = 16 + t
        seam = np.abs(outa[:, seam_x - 1] - outa[:, seam_x]).mean()
        adj = [np.abs(outa[:, x] - outa[:, x + 1]).mean()
               for x in range(seam_x, seam_x + K - 1)]
        check("seam step within texture", seam <= 1.5 * max(np.median(adj), 1.0),
              f"seam {seam:.1f} vs band adj {np.median(adj):.1f}")

        # I/J: JPEG fixture — coefficient identity + decoded border sanity
        import jpeglib
        jp = tmp / "real.jpg"
        Image.open(real_p).convert("RGB").save(jp, quality=88)
        o = process_file(jp, _mk_args(base, extend="16"))
        d0, d1 = jpeglib.read_dct(str(jp)), jpeglib.read_dct(str(o))
        okc = bool(np.array_equal(np.asarray(d0.qt), np.asarray(d1.qt)))
        ext_j = resolve_extents(_mk_args(base, extend="16"),
                                (d0.width, d0.height), jpeg_mcu(d0), [])
        for ci, (a0, a1) in enumerate(zip([d0.Y, d0.Cb, d0.Cr],
                                          [d1.Y, d1.Cb, d1.Cr])):
            rows, cols, ov, oh = jpeg_paste_box(d0, ci, (d0.width, d0.height),
                                                ext_j, False)
            okc &= bool(np.array_equal(
                a1[rows, cols],
                a0[rows.start - ov:rows.stop - ov, cols.start - oh:cols.stop - oh]))
        check("JPEG interior coefficients bit-exact", okc,
              f"extents {ext_j}")
        p0 = jpeglib.read_spatial(str(jp)).spatial.astype(np.float32)
        p1 = jpeglib.read_spatial(str(o)).spatial.astype(np.float32)
        bmean0 = p0[100:-100, 2:4].mean(axis=(0, 1))   # near the de-trend anchor
        bmean1 = p1[100:-100, :12].mean(axis=(0, 1))
        check("JPEG synth border color sane",
              float(np.abs(bmean0 - bmean1).max()) <= 10.0,
              f"delta {np.abs(bmean0 - bmean1).max():.1f}")

        # J2: 4:2:2 JPEG (asymmetric subsampling) — regression for the
        # samp_factor axis order; must not crash and must stay bit-exact
        jp422 = tmp / "real422.jpg"
        Image.open(real_p).convert("RGB").save(jp422, quality=90, subsampling=1)
        try:
            o = process_file(jp422, _mk_args(base, extend="16"))
            d0, d1 = jpeglib.read_dct(str(jp422)), jpeglib.read_dct(str(o))
            ext_j = resolve_extents(_mk_args(base, extend="16"),
                                    (d0.width, d0.height), jpeg_mcu(d0), [])
            ok422 = True
            for ci, (a0, a1) in enumerate(zip([d0.Y, d0.Cb, d0.Cr],
                                              [d1.Y, d1.Cb, d1.Cr])):
                rows, cols, ov, oh = jpeg_paste_box(
                    d0, ci, (d0.width, d0.height), ext_j, False)
                ok422 &= bool(np.array_equal(
                    a1[rows, cols],
                    a0[rows.start - ov:rows.stop - ov,
                       cols.start - oh:cols.stop - oh]))
            check("JPEG 4:2:2 interior coefficients bit-exact", ok422,
                  f"extents {ext_j}, mcu {jpeg_mcu(d0)}")
        except Exception as e:
            check("JPEG 4:2:2 interior coefficients bit-exact", False,
                  f"{type(e).__name__}: {e}")

        # K: WebP round trip identity
        wp = tmp / "real.webp"
        Image.open(real_p).convert("RGB").save(wp, quality=80)
        src_w = np.asarray(Image.open(wp).convert("RGB"))
        o = process_file(wp, _mk_args(base, extend="16"))
        outw = np.asarray(Image.open(o).convert("RGB"))
        t = TRIM_CAP
        check("interior identity WebP",
              np.array_equal(outw[16 + t:16 + src_w.shape[0] - t,
                                  16 + t:16 + src_w.shape[1] - t],
                             src_w[t:-t, t:-t]))
    else:
        console.print("  [yellow](no real PNG input found; "
                      "skipped scan-based checks)[/]")

    # L: flat border stays clean (both modes)
    for mode in ("smart", "naive"):
        o = process_file(flat_p, _mk_args(base, extend="16", mode=mode,
                                          suffix=f"_ext_{mode}"))
        fa = np.asarray(Image.open(o)).astype(np.int16)
        dev = np.abs(fa[100:-100, :16] - np.array([255, 214, 0])).max()
        check(f"flat border clean ({mode})", dev <= 14, f"max dev {dev}")

    # M: aspect fill
    wide_p = tmp / "wide.png"
    Image.fromarray(bordered(340, 420, (200, 200, 60), (255, 255, 255))).save(wide_p)
    o = process_file(wide_p, _mk_args(base, extend="0", fix_aspect=True))
    w, h = Image.open(o).size
    check("aspect fill (too wide -> pad height)",
          w == 340 and h == round(340 * 88 / 63), f"{w}x{h}")

    # N: speckle statistics on synthetic speckle fixture
    o = process_file(speck_p, _mk_args(base, extend="16"))
    sa = np.asarray(Image.open(o)).astype(np.float32)
    rs = highpass_std(sa[100:-100, :16]).mean()
    rb = highpass_std(sa[100:-100, 16:23]).mean()
    check("speckle grain reproduced", 0.5 <= rs / max(rb, 1e-6) <= 1.6,
          f"ratio {rs / max(rb, 1e-6):.2f}")

    fails = [r for r in results if not r[1]]
    color = "red" if fails else "green"
    console.print(f"\n[bold {color}]selfcheck: "
                  f"{len(results) - len(fails)}/{len(results)} passed[/]")
    return 1 if fails else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.STYLE_OPTIONS_TABLE_LEADING = 0
_GROUPS = [
    {"name": "Sizing",
     "options": ["--extend", "--left", "--right", "--top", "--bottom",
                 "--target", "--fix-aspect", "--card-size"]},
    {"name": "Synthesis",
     "options": ["--mode", "--sample", "--trim", "--jitter", "--jitter-smooth",
                 "--jitter-cross", "--noise", "--smudge", "--seam-feather",
                 "--corner-guard", "--halo", "--seed"]},
    {"name": "Input / Output",
     "options": ["--out-dir", "--suffix", "--compare", "--force",
                 "--recursive", "--dry-run", "--help"]},
]
click.rich_click.OPTION_GROUPS = {"*": _GROUPS}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("inputs", nargs=-1, metavar="[INPUTS]...")
# sizing ---------------------------------------------------------------------
@click.option("-e", "--extend", default="16", show_default=True, metavar="AMT",
              help="Border to add per edge: px ([cyan]16[/]) or mm ([cyan]2.5mm[/]).")
@click.option("--left", default=None, metavar="AMT",
              help="Override for the left edge ([cyan]0[/] skips it).")
@click.option("--right", default=None, metavar="AMT",
              help="Override for the right edge.")
@click.option("--top", default=None, metavar="AMT",
              help="Override for the top edge.")
@click.option("--bottom", default=None, metavar="AMT",
              help="Override for the bottom edge.")
@click.option("--target", default=None, metavar="WxH",
              help="Pad (centered) to an exact final size, e.g. "
                   "[cyan]69x94mm[/] or [cyan]440x600[/]; overrides -e.")
@click.option("--fix-aspect", is_flag=True,
              help="First pad the short axis so the image matches the card "
                   "aspect ratio exactly, then add the border.")
@click.option("--card-size", default="63x88", show_default=True, metavar="WxH",
              help="Physical card size in mm — basis for all mm math "
                   "(embedded file DPI is never trusted).")
# synthesis ------------------------------------------------------------------
@click.option("--mode", type=click.Choice(["smart", "naive"]), default="smart",
              show_default=True,
              help="[cyan]smart[/]: jittered band resampling · "
                   "[cyan]naive[/]: replicate the outermost clean line "
                   "(plus noise + smudge).")
@click.option("-k", "--sample", type=int, default=8, show_default=True,
              metavar="N",
              help="How many border pixels to sample patterns/colors from "
                   "(auto-clamped before inner border structure).")
@click.option("--trim", default="auto", show_default=True, metavar="auto|N",
              help="Outermost pixels treated as scanner bloom / cut-line junk: "
                   "excluded from sampling and (png/webp) replaced. "
                   "[cyan]auto[/] detects hard bloom lines per edge (max 3).")
@click.option("--jitter", type=float, default=0.85, show_default=True,
              help="0..1 randomness of the sampling depth "
                   "([cyan]0[/] = plain pattern continuation).")
@click.option("--jitter-smooth", type=float, default=1.2, show_default=True,
              metavar="SIGMA",
              help="Smoothing of the jitter field; matches speckle grain size "
                   "([cyan]0[/] = per-pixel salt & pepper).")
@click.option("--jitter-cross", type=float, default=4.0, show_default=True,
              metavar="PX",
              help="Along-edge wobble of the sampling position; kills "
                   "repeated-fleck trails ([cyan]0[/] = perfectly straight).")
@click.option("--noise", type=float, default=0.35, show_default=True,
              metavar="F",
              help="Added grain as a multiple of the border's own measured "
                   "grain (self-tuning; [cyan]0[/] = off).")
@click.option("--smudge", type=float, default=0.6, show_default=True,
              metavar="SIGMA",
              help="Gaussian smudge of the new border, ramped toward the "
                   "outer edge ([cyan]0[/] = off).")
@click.option("--seam-feather", type=int, default=3, show_default=True,
              metavar="PX",
              help="Pixels over which randomness ramps in from the seam.")
@click.option("--corner-guard", type=int, default=12, show_default=True,
              metavar="PX",
              help="Keep sampling this far away from image corners (avoids "
                   "seeding from rounded/white scan corners).")
@click.option("--halo", type=click.Choice(["auto", "overwrite", "blend"]),
              default="auto", show_default=True,
              help="Trimmed halo ring handling: [cyan]overwrite[/] it "
                   "(png/webp default) or [cyan]blend[/] it out (jpeg "
                   "default; overwrite on jpeg re-encodes the outer "
                   "block ring).")
@click.option("--seed", type=int, default=0, show_default=True,
              help="RNG seed (per-file streams derived from filename).")
# input/output ---------------------------------------------------------------
@click.option("-o", "--out-dir", default=None, metavar="DIR",
              help="Output directory (default: alongside each input).")
@click.option("--suffix", default="_ext", show_default=True,
              help="Appended to the output file stem.")
@click.option("--compare", is_flag=True,
              help="Also write a QA sheet: original | result | result with "
                   "the original boundary marked.")
@click.option("--force", is_flag=True,
              help="Overwrite existing output files "
                   "(inputs are never overwritten).")
@click.option("--recursive", is_flag=True,
              help="Descend into subdirectories.")
@click.option("--dry-run", is_flag=True,
              help="Show what would be done without writing anything.")
@click.option("--selfcheck", is_flag=True, hidden=True)
@click.version_option(__version__, "-V", "--version")
@click.pass_context
def cli(ctx: click.Context, **kw) -> None:
    """Extend card scan borders for printing, continuing the existing
    border pattern (holo speckle, solid colors, ...).

    Original image data is [bold]never re-encoded[/]: PNG/WebP pixels stay
    bit-identical and JPEG goes through lossless DCT-block surgery.
    INPUTS are image files and/or directories (png/jpg/jpeg/webp).

    [dim]Examples:[/]

    [dim]  cardbleed card.png --compare[/]

    [dim]  cardbleed ./cards/ -e 2.5mm --fix-aspect[/]
    """
    ctx.exit(run(SimpleNamespace(**kw)))


def run(args) -> int:
    console = Console(highlight=False)
    err = Console(stderr=True, highlight=False)
    if args.selfcheck:
        return selfcheck(args)
    if not args.inputs:
        raise click.UsageError("no inputs given (image files or directories)")

    files, input_errors = iter_inputs(list(args.inputs), args.recursive,
                                      args.suffix)
    for e in input_errors:
        err.print(f"[yellow]SKIPPED[/] {e}")
    if not files:
        err.print("[bold red]error:[/] no supported images found")
        return 2

    ok, failed = 0, len(input_errors)
    claimed: dict = {}
    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}"),
                  BarColumn(), MofNCompleteColumn(),
                  console=console, transient=True,
                  disable=len(files) < 3 or args.dry_run) as progress:
        task = progress.add_task("extending", total=len(files))
        for f in files:
            progress.update(task, description=f.name)
            try:
                process_file(f, args, console=console, claimed=claimed)
                ok += 1
            except FileError as e:
                err.print(f"[bold cyan]{f.name}[/]: [yellow]SKIPPED[/] — {e}")
                failed += 1
            except Exception as e:  # unexpected: report, keep the batch alive
                err.print(f"[bold cyan]{f.name}[/]: [bold red]ERROR[/] — "
                          f"{type(e).__name__}: {e}")
                failed += 1
            progress.advance(task)
    if len(files) > 1:
        parts = [f"[green]{ok} ok[/]"]
        if failed:
            parts.append(f"[red]{failed} failed[/]")
        console.print(f"[bold]done:[/] " + ", ".join(parts))
    return 1 if failed else 0


if __name__ == "__main__":
    cli()
