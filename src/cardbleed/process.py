"""Per-file processing pipeline and batch input handling.

Order of operations: load → square corners → fit (stretch/crop to card aspect
with target borders) → cut bleed → synthesize the added border → save. The
lossless guarantee (PNG bit-identical, JPEG DCT-copy) holds when the original
pixels are untouched — i.e. no stretch and no crop; those paths re-encode.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console

from .corners import square_background
from .errors import FileError
from .formats import FORMATS, jpeg_mcu, load_pixels, save_jpeg, save_png_webp
from .geometry import EDGES, Edges, parse_size, resolve_bleed, solve_fit
from .synthesis import TRIM_CAP, Params, extend_image

MAGENTA = np.array([255, 0, 255], dtype=np.uint8)


def iter_inputs(
    paths: list[str], recursive: bool, suffix: str
) -> tuple[list[Path], list[str]]:
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


def make_compare(
    orig: np.ndarray, result: np.ndarray, extents: tuple[int, int, int, int]
) -> np.ndarray:
    def rgb(a):
        if a.shape[2] < 3:
            return np.repeat(a[:, :, :1], 3, axis=2)
        return a[:, :, :3]

    o, r = rgb(orig), rgb(result)
    marked = r.copy()
    left, top, _, _ = extents
    h0, w0 = o.shape[:2]
    marked[top, left : left + w0] = MAGENTA
    marked[top + h0 - 1, left : left + w0] = MAGENTA
    marked[top : top + h0, left] = MAGENTA
    marked[top : top + h0, left + w0 - 1] = MAGENTA

    gut, pad = 12, 12
    hmax = max(o.shape[0], r.shape[0])
    wsum = o.shape[1] + r.shape[1] + marked.shape[1] + 2 * gut + 2 * pad
    sheet = np.full((hmax + 2 * pad, wsum, 3), 96, dtype=np.uint8)
    x = pad
    for panel in (o, r, marked):
        sheet[pad : pad + panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1] + gut
    return sheet


def _mcu_align(
    a: int, b: int, m: int, notes: list[str], lo: str, hi: str
) -> tuple[int, int]:
    """Snap a JPEG edge extension to a whole MCU, moving the remainder to the
    opposite edge so the final size stays exact."""
    if a % m == 0:
        return a, b
    snapped = round(a / m) * m
    if snapped > a + b:
        snapped = (a // m) * m
    notes.append(
        f"JPEG alignment: {lo} {a}->{snapped}px (offset must be a multiple of "
        f"{m}; difference moved to {hi})"
    )
    return snapped, b + (a - snapped)


def _edge_args(args, prefix: str) -> dict[str, str | None]:
    return {e: getattr(args, f"{prefix}_{e}", None) for e in EDGES}


def process_file(
    path: Path, args, console: Console | None = None, claimed: dict | None = None
) -> Path:
    fmt = FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise FileError(f"unsupported format {path.suffix!r} (png/jpg/jpeg/webp only)")
    notes: list[str] = []
    arr, meta = load_pixels(path, fmt, notes)
    h0, w0 = arr.shape[:2]
    card_w, card_h, _ = parse_size(args.card_size)

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else path.parent
    out = out_dir / f"{path.stem}{args.suffix}{path.suffix.lower()}"
    cmp_path = out_dir / f"{path.stem}{args.suffix}_compare.png"
    for target in (out, cmp_path) if args.compare else (out,):
        if target.exists() and target.resolve() == path.resolve():
            raise FileError(
                f"output {target.name} equals the input (set --suffix/--out-dir)"
            )
    if claimed is not None:
        key = str(out.resolve()) if out.is_absolute() else str(out)
        if key in claimed:
            raise FileError(
                f"output {out.name} already produced by {claimed[key]} this batch"
            )
        claimed[key] = path.name

    # --- square rounded/ragged corners (before geometry) ---------------------
    if args.fill_corners:
        if fmt == "jpeg":
            notes.append("fill-corners skipped: JPEG splices original blocks")
        else:
            arr = square_background(arr, notes)
            h0, w0 = arr.shape[:2]

    # --- geometry: fit (optional) + cut bleed --------------------------------
    tgt_over = _edge_args(args, "border_target")
    fit = None
    if args.border_target or any(tgt_over.values()):
        current = Edges.from_edges(None, **_edge_args(args, "border_current"))
        target = Edges.from_edges(args.border_target, **tgt_over)
        fit = solve_fit(
            w0,
            h0,
            current,
            target,
            card_w,
            card_h,
            stretch=args.stretch,
            crop=args.crop,
        )
    trim_w = fit.trim_w if fit else float(w0)
    trim_h = fit.trim_h if fit else float(h0)
    bleed_over = _edge_args(args, "bleed")
    bleed_edges = (
        Edges.from_edges(args.bleed, **bleed_over)
        if (args.bleed or any(bleed_over.values()))
        else None
    )
    bleed_px = resolve_bleed(bleed_edges, trim_w, card_w, trim_h, card_h)

    sx, sy = (fit.stretch_x, fit.stretch_y) if fit else (1.0, 1.0)
    crop = {e: max(0.0, -(fit.ext[e] if fit else 0.0)) for e in EDGES}
    modified = fit is not None and (
        abs(sx - 1) > 1e-9 or abs(sy - 1) > 1e-9 or any(crop[e] > 0.5 for e in EDGES)
    )

    # apply stretch (resample) -----------------------------------------------
    if abs(sx - 1) > 1e-9 or abs(sy - 1) > 1e-9:
        nw, nh = max(1, round(w0 * sx)), max(1, round(h0 * sy))
        mode = meta.get("mode")
        pim = Image.fromarray(
            arr[:, :, 0] if arr.shape[2] == 1 else arr,
            mode=mode if isinstance(mode, str) else "RGB",
        )
        arr = np.asarray(pim.resize((nw, nh), Image.Resampling.LANCZOS))
        if arr.ndim == 2:
            arr = arr[:, :, None]
        h0, w0 = arr.shape[:2]
        notes.append(f"stretched x{sx:.3f} y{sy:.3f} to hit target borders")

    # apply crop --------------------------------------------------------------
    cl, cr, ct, cb = (round(crop[e]) for e in ("left", "right", "top", "bottom"))
    if cl or cr or ct or cb:
        arr = arr[ct : h0 - cb, cl : w0 - cr]
        h0, w0 = arr.shape[:2]
        notes.append(f"shaved border -L{cl} -R{cr} -T{ct} -B{cb}px (over target)")

    # fit pad → an exact-aspect integer trim (derive one axis from the other so
    # the card lands on 63:88 to the pixel, not per-edge-rounded), then cut bleed.
    if fit:
        fp = {e: max(0.0, fit.ext[e]) for e in EDGES}
        ca = card_w / card_h
        # closest integer (tw, th) to the exact card aspect, in a ±1 window
        cands = [
            (max(w0, round(th * ca)), th)
            for th in (round(fit.trim_h) - 1, round(fit.trim_h), round(fit.trim_h) + 1)
            if th >= h0
        ]
        tw, th = min(cands, key=lambda wh: abs(wh[0] / wh[1] - ca))
        trim_w, trim_h = float(tw), float(th)
        wsum, hsum = fp["left"] + fp["right"], fp["top"] + fp["bottom"]
        fl = round((tw - w0) * fp["left"] / wsum) if wsum > 1e-9 else (tw - w0) // 2
        ft = round((th - h0) * fp["top"] / hsum) if hsum > 1e-9 else (th - h0) // 2
        fpad = {"left": fl, "right": tw - w0 - fl, "top": ft, "bottom": th - h0 - ft}
    else:
        fpad = dict.fromkeys(EDGES, 0)
    bl = {e: round(bleed_px[e]) for e in EDGES}
    left, top, right, bottom = (
        fpad[e] + bl[e] for e in ("left", "top", "right", "bottom")
    )

    halo = (
        args.halo
        if args.halo != "auto"
        else ("blend" if fmt == "jpeg" else "overwrite")
    )
    overwrite = halo == "overwrite"
    if fmt == "jpeg" and overwrite:
        notes.append("halo overwrite on JPEG: outer block ring re-encoded")

    if fmt == "jpeg" and not modified and (left or top or right or bottom):
        mcu = jpeg_mcu(meta["dct"])
        left, right = _mcu_align(left, right, mcu[0], notes, "left", "right")
        top, bottom = _mcu_align(top, bottom, mcu[1], notes, "top", "bottom")
    extents = (left, top, right, bottom)

    if fit:
        b = fit.borders
        notes.append(
            f"fit → {trim_w:.0f}x{trim_h:.0f}px, borders "
            f"L{b['left'] * 100:.1f} R{b['right'] * 100:.1f} "
            f"T{b['top'] * 100:.1f} B{b['bottom'] * 100:.1f}%"
        )
        if not args.stretch and abs(fit.over_ratio) > 0.005:
            side = "wide" if fit.over_ratio > 0 else "tall"
            notes.append(
                f"art ~{abs(fit.over_ratio) * 100:.1f}% too {side} vs target; "
                "use --stretch for exact borders"
            )

    if not any(extents) and not modified:
        raise FileError("nothing to do (no fit change and no bleed)")

    if args.dry_run:
        if console is not None:
            console.print(
                f"[bold cyan]{path.name}[/] {w0}x{h0} "
                f"+(L{left} T{top} R{right} B{bottom}) → "
                f"[bold]{w0 + left + right}x{h0 + top + bottom}[/]"
            )
            for n in dict.fromkeys(notes):
                console.print(f"   [dim]• {n}[/]")
        return out
    if out.exists() and not args.force:
        raise FileError(f"{out.name} exists (use --force to overwrite outputs)")

    # --- synthesize the added border -----------------------------------------
    p = Params(
        mode=args.mode,
        sample=args.sample,
        trim=args.trim,
        jitter=args.jitter,
        shuffle=args.shuffle,
        noise=args.noise,
        smudge=args.smudge,
        seam_feather=args.seam_feather,
        halo=args.halo,
        edge_fill=args.edge_fill,
    )
    rng = np.random.default_rng([args.seed, zlib.crc32(path.name.encode())])
    result = (
        extend_image(arr, extents, p, rng, overwrite, notes) if any(extents) else arr
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = (trim_w / card_w * 25.4, trim_h / card_h * 25.4)
    if fmt == "jpeg" and not modified:
        trim_px = TRIM_CAP if args.trim == "auto" else int(args.trim)
        save_jpeg(result, meta, out, (w0, h0), extents, overwrite, trim_px)
    elif fmt == "jpeg":
        rgb = result[:, :, 0] if result.shape[2] == 1 else result[:, :, :3]
        Image.fromarray(rgb).save(out, format="JPEG", quality=95, subsampling=0)
        notes.append(
            "JPEG re-encoded (stretch/crop can't reuse the original DCT blocks)"
        )
    else:
        save_png_webp(result, meta, out, fmt, dpi)

    if args.compare:
        sheet = make_compare(arr, result, extents)
        Image.fromarray(sheet).save(cmp_path)
        notes.append(f"comparison sheet: {cmp_path.name}")

    if console is not None:
        console.print(
            f"[bold cyan]{path.name}[/] {w0}×{h0} → "
            f"[bold]{result.shape[1]}×{result.shape[0]}[/] "
            f"[dim](· {args.mode})[/] → [green]{out.name}[/]"
        )
        for n in dict.fromkeys(notes):
            console.print(f"   [dim]• {n}[/]")
    return out
