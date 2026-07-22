"""Per-file processing pipeline and batch input handling."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console

from .errors import FileError
from .formats import FORMATS, jpeg_mcu, load_pixels, save_jpeg, save_png_webp
from .sizing import parse_size, resolve_extents
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
        if a.shape[2] < 3:  # L / LA: show the gray channel
            return np.repeat(a[:, :, :1], 3, axis=2)
        return a[:, :, :3]

    o, r = rgb(orig), rgb(result)
    marked = r.copy()
    L, T, _, _ = extents
    h0, w0 = o.shape[:2]
    marked[T, L : L + w0] = MAGENTA
    marked[T + h0 - 1, L : L + w0] = MAGENTA
    marked[T : T + h0, L] = MAGENTA
    marked[T : T + h0, L + w0 - 1] = MAGENTA

    gut, pad = 12, 12
    hmax = max(o.shape[0], r.shape[0])
    wsum = o.shape[1] + r.shape[1] + marked.shape[1] + 2 * gut + 2 * pad
    sheet = np.full((hmax + 2 * pad, wsum, 3), 96, dtype=np.uint8)
    x = pad
    for panel in (o, r, marked):
        sheet[pad : pad + panel.shape[0], x : x + panel.shape[1]] = panel
        x += panel.shape[1] + gut
    return sheet


def process_file(
    path: Path, args, console: Console | None = None, claimed: dict | None = None
) -> Path:
    fmt = FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise FileError(f"unsupported format {path.suffix!r} (png/jpg/jpeg/webp only)")
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
            raise FileError(
                f"output path {target.name} equals the input; "
                "refusing to overwrite the source "
                "(set --suffix or --out-dir)"
            )
    if claimed is not None:
        key = str(out.resolve()) if out.is_absolute() else str(out)
        if key in claimed:
            raise FileError(
                f"output {out.name} was already produced by "
                f"{claimed[key]} in this batch (same stem); "
                "use --out-dir per folder or rename inputs"
            )
        claimed[key] = path.name
    if not any(extents):
        raise FileError("nothing to do (all extension amounts are 0)")

    if args.dry_run:
        if console is not None:
            clash = (
                " [red](exists: needs --force)[/]"
                if out.exists() and not args.force
                else ""
            )
            console.print(
                f"[bold cyan]{path.name}[/] {W0}×{H0} → "
                f"[bold]{W0 + eL + eR}×{H0 + eT + eB}[/] "
                f"[dim](+L{eL} +T{eT} +R{eR} +B{eB})[/] → {out}{clash}"
            )
            for n in dict.fromkeys(notes):
                console.print(f"   [dim]• {n}[/]")
        return out
    if out.exists() and not args.force:
        raise FileError(f"{out.name} exists (use --force to overwrite outputs)")

    p = Params(
        mode=args.mode,
        sample=args.sample,
        trim=args.trim,
        jitter=args.jitter,
        jitter_smooth=args.jitter_smooth,
        jitter_cross=args.jitter_cross,
        shuffle=args.shuffle,
        noise=args.noise,
        smudge=args.smudge,
        seam_feather=args.seam_feather,
        corner_guard=args.corner_guard,
        halo=args.halo,
        edge_fill=args.edge_fill,
    )
    halo = p.halo if p.halo != "auto" else ("blend" if fmt == "jpeg" else "overwrite")
    overwrite = halo == "overwrite"
    if fmt == "jpeg" and overwrite:
        notes.append(
            "halo overwrite on JPEG: the outer block ring is "
            "re-encoded (localized loss inside the border)"
        )

    rng = np.random.default_rng([args.seed, zlib.crc32(path.name.encode())])
    result = extend_image(arr, extents, p, rng, overwrite, notes)

    out_dir.mkdir(parents=True, exist_ok=True)
    card_w, card_h, _ = parse_size(args.card_size)
    dpi = (W0 / card_w * 25.4, H0 / card_h * 25.4)
    if args.fix_aspect:
        ref = dpi[0] if card_w / card_h < W0 / H0 else dpi[1]
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
            f" → [green]{out.name}[/]"
        )
        for n in dict.fromkeys(notes):  # dedupe, keep order
            console.print(f"   [dim]• {n}[/]")
    return out
