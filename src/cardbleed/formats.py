"""Format-preserving image I/O.

The whole point of cardbleed is that original image data is never re-encoded:

  PNG  -> PNG   lossless container; original pixels bit-identical
  WebP -> WebP  written lossless (exact); decoded original pixels preserved
  JPEG -> JPEG  DCT-domain surgery: original quantized coefficient blocks are
                copied bit-exact into a larger grid; only new border blocks
                are encoded, using the file's own quantization tables

Adding a new format means adding a loader + saver here; synthesis is
format-agnostic (it only ever sees an (H, W, C) uint8 array).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import FileError
from .synthesis import TRIM_CAP

FORMATS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}


# --------------------------------------------------------------------------
# loading
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


# --------------------------------------------------------------------------
# PNG / WebP saving (lossless)
# --------------------------------------------------------------------------

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
