"""Built-in assertion suite (`cardbleed --selfcheck [scan.png ...]`).

Covers geometry, the border-fit guarantee (exact card aspect), format identity
(PNG/WebP/JPEG lossless on the extend path), determinism, and texture stats —
plus deeper scan-based checks when a real PNG scan is passed.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from rich.console import Console

from .filters import box_blur3, highpass_std
from .process import _mcu_align, process_file
from .synthesis import TRIM_CAP, Params, edge_geometry

_CARD = 63 / 88


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

    tmp = Path(tempfile.mkdtemp(prefix="cardbleed_check_"))
    console.print(f"[dim]selfcheck workspace: {tmp}[/]")
    base = _mk_args(
        args,
        out_dir=str(tmp / "out"),
        force=True,
        compare=False,
        dry_run=False,
        suffix="_ext",
    )

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
    Image.fromarray(
        bordered(300, 420, (180, 150, 90), (120, 120, 120), noise_sigma=25)
    ).save(speck_p)
    grad_p = tmp / "grad.png"
    Image.fromarray(
        (np.linspace(30, 220, 48)[:, None, None] * np.ones((48, 64, 3))).astype(
            np.uint8
        )
    ).save(grad_p)

    real_p = None
    for cand in args.inputs or []:
        c = Path(cand).expanduser()
        if c.is_file() and c.suffix.lower() == ".png":
            real_p = c
            break

    # A: geometry, uniform bleed in px
    o = process_file(flat_p, _mk_args(base, bleed="16px"))
    check(
        "bleed px (+16 all)", Image.open(o).size == (332, 452), f"{Image.open(o).size}"
    )

    # B: orientation — bleed top only
    o = process_file(grad_p, _mk_args(base, bleed_top="8px", trim="0", suffix="_top"))
    ga = np.asarray(Image.open(o))
    check(
        "orientation (top-only bleed)",
        ga.shape[:2] == (56, 64)
        and np.array_equal(ga[8:], np.asarray(Image.open(grad_p))),
        f"shape {ga.shape[:2]}",
    )

    # C: fit lands exactly on the card aspect (a card-realistic, mildly-too-wide
    # fixture — ~2% off, like a real scrydex scan)
    wide = tmp / "fitwide.png"
    Image.fromarray(bordered(600, 810, (200, 200, 60), (255, 255, 255), bw=30)).save(
        wide
    )
    common = dict(
        border_current_top="30px",
        border_current_right="30px",
        border_current_bottom="30px",
        border_current_left="30px",
    )
    o = process_file(wide, _mk_args(base, border_target="5%", suffix="_fit", **common))
    w, h = Image.open(o).size
    check(
        "fit aspect exact (no stretch)",
        abs(w / h - _CARD) < 2e-3,
        f"{w}x{h} = {w / h:.4f}",
    )
    o = process_file(
        wide, _mk_args(base, border_target="5%", stretch=True, suffix="_fits", **common)
    )
    w, h = Image.open(o).size
    check(
        "fit aspect exact (stretch)",
        abs(w / h - _CARD) < 1.5e-3,
        f"{w}x{h} = {w / h:.4f}",
    )

    # D: crop an over-target border (Variant C)
    fatb = tmp / "fatbottom.png"
    a = bordered(300, 420, (255, 214, 0), (255, 255, 255), bw=18)
    a = np.pad(a, ((0, 80), (0, 0), (0, 0)), mode="edge")  # thick bottom border
    Image.fromarray(a).save(fatb)
    o = process_file(
        fatb,
        _mk_args(
            base,
            border_target="5%",
            suffix="_crop",
            border_current_top="18px",
            border_current_right="18px",
            border_current_bottom="98px",
            border_current_left="18px",
        ),
    )
    w, h = Image.open(o).size
    check("crop over-target edge → aspect exact", abs(w / h - _CARD) < 2e-3, f"{w}x{h}")

    # E: flat border stays clean (both modes)
    for mode in ("smart", "naive"):
        o = process_file(
            flat_p, _mk_args(base, bleed="16px", mode=mode, suffix=f"_{mode}")
        )
        dev = np.abs(
            np.asarray(Image.open(o)).astype(np.int16)[100:-100, :16] - [255, 214, 0]
        ).max()
        check(f"flat border clean ({mode})", dev <= 14, f"max dev {dev}")

    # F: speckle grain reproduced
    o = process_file(speck_p, _mk_args(base, bleed="16px", suffix="_spk"))
    sa = np.asarray(Image.open(o)).astype(np.float32)
    ratio = highpass_std(sa[100:-100, :16]).mean() / max(
        highpass_std(sa[100:-100, 16:23]).mean(), 1e-6
    )
    check("speckle grain reproduced", 0.5 <= ratio <= 1.6, f"ratio {ratio:.2f}")

    # G: pattern mode continues a periodic lattice exactly
    lat_p = tmp / "lattice.png"
    la = np.full((420, 300, 3), (60, 110, 120), dtype=np.uint8)
    yy, xx = np.mgrid[0:420, 0:300]
    la[(yy % 6 < 2) & (xx % 6 < 2)] = (220, 235, 240)
    la[24:-24, 24:-24] = (250, 250, 250)
    Image.fromarray(la).save(lat_p)
    o = process_file(
        lat_p,
        _mk_args(
            base,
            bleed="16px",
            mode="pattern",
            trim="0",
            noise=0.0,
            smudge=0.0,
            suffix="_pat",
        ),
    )
    lo = np.asarray(Image.open(o)).astype(np.int16)[100:-100]
    dev = max(
        int(np.abs(lo[:, x] - lo[:, 16 + ((x - 16) % 6)]).max()) for x in range(16)
    )
    check("pattern mode continues lattice", dev <= 4, f"max dev {dev}")

    if real_p is None:
        console.print("  [yellow](no real PNG input; skipped scan-based checks)[/]")
    else:
        src = np.asarray(Image.open(real_p).convert("RGB"))
        W0, H0 = src.shape[1], src.shape[0]
        t = TRIM_CAP

        o = process_file(real_p, _mk_args(base, bleed="16px"))
        outa = np.asarray(Image.open(o))
        crop = outa[16 + t : 16 + H0 - t, 16 + t : 16 + W0 - t]
        check("interior identity PNG", np.array_equal(crop, src[t:-t, t:-t]))

        o2 = process_file(real_p, _mk_args(base, bleed="16px"))
        check("determinism (same seed)", o.read_bytes() == o2.read_bytes())

        o = process_file(real_p, _mk_args(base, bleed="2mm", suffix="_mm"))
        exp = (W0 + 2 * round(2 * W0 / 63), H0 + 2 * round(2 * H0 / 88))
        check("bleed mm", Image.open(o).size == exp, f"{Image.open(o).size} vs {exp}")

        o = process_file(real_p, _mk_args(base, bleed="16px", trim="1", suffix="_t1"))
        outa = np.asarray(Image.open(o)).astype(np.float32)
        synth, band = outa[100:-100, :16], outa[100:-100, 17:24]
        ratio = highpass_std(synth).mean() / max(highpass_std(band).mean(), 1e-6)
        check("texture stats (residual std ratio)", 0.5 <= ratio <= 1.6, f"{ratio:.2f}")
        mean_d = abs(synth.mean(axis=(0, 1)) - band.mean(axis=(0, 1))).max()
        check("texture mean vs band (<=8/255)", bool(mean_d <= 8.0), f"{mean_d:.1f}")

        hp = synth - box_blur3(synth)
        hb = band - box_blur3(band)

        def lag3(x):
            aa, bb = x[:, :-3].ravel(), x[:, 3:].ravel()
            aa, bb = aa - aa.mean(), bb - bb.mean()
            return float((aa * bb).sum()) / max(
                math.sqrt(float((aa * aa).sum() * (bb * bb).sum())), 1e-6
            )

        check(
            "no streaks (lag-3 autocorr)",
            lag3(hp.mean(axis=2)) <= max(1.5 * abs(lag3(hb.mean(axis=2))), 0.35),
            f"{lag3(hp.mean(axis=2)):.2f} vs {lag3(hb.mean(axis=2)):.2f}",
        )

        # JPEG interior coefficients bit-exact (extend path, DCT surgery)
        import jpeglib

        from .formats import jpeg_mcu, jpeg_paste_box

        jp = tmp / "real.jpg"
        Image.open(real_p).convert("RGB").save(jp, quality=88)
        o = process_file(jp, _mk_args(base, bleed="16px"))
        d0, d1 = jpeglib.read_dct(str(jp)), jpeglib.read_dct(str(o))
        mcu = jpeg_mcu(d0)
        eL, eR = _mcu_align(16, 16, mcu[0], [], "l", "r")
        eT, eB = _mcu_align(16, 16, mcu[1], [], "t", "b")
        ext_j = (eL, eT, eR, eB)
        okc = bool(np.array_equal(np.asarray(d0.qt), np.asarray(d1.qt)))
        for ci, (a0, a1) in enumerate(
            zip([d0.Y, d0.Cb, d0.Cr], [d1.Y, d1.Cb, d1.Cr], strict=True)
        ):
            rows, cols, ov, oh = jpeg_paste_box(
                d0, ci, (d0.width, d0.height), ext_j, False
            )
            okc &= bool(
                np.array_equal(
                    a1[rows, cols],
                    a0[
                        rows.start - ov : rows.stop - ov,
                        cols.start - oh : cols.stop - oh,
                    ],
                )
            )
        check("JPEG interior coefficients bit-exact", okc, f"extents {ext_j}")
        p0 = np.asarray(jpeglib.read_spatial(str(jp)).spatial, dtype=np.float32)
        p1 = np.asarray(jpeglib.read_spatial(str(o)).spatial, dtype=np.float32)
        tj, kj = edge_geometry(np.ascontiguousarray(p0), Params(sample=base.sample))[:2]
        d = float(
            np.abs(
                p0[100:-100, tj : tj + kj].mean(axis=(0, 1))
                - p1[100:-100, :12].mean(axis=(0, 1))
            ).max()
        )
        check("JPEG synth border color sane", d <= 10.0, f"delta {d:.1f}")

        # WebP round-trip identity
        wp = tmp / "real.webp"
        Image.open(real_p).convert("RGB").save(wp, quality=80)
        src_w = np.asarray(Image.open(wp).convert("RGB"))
        o = process_file(wp, _mk_args(base, bleed="16px"))
        outw = np.asarray(Image.open(o).convert("RGB"))
        check(
            "interior identity WebP",
            np.array_equal(
                outw[
                    16 + t : 16 + src_w.shape[0] - t, 16 + t : 16 + src_w.shape[1] - t
                ],
                src_w[t:-t, t:-t],
            ),
        )

    fails = [r for r in results if not r[1]]
    color = "red" if fails else "green"
    passed = len(results) - len(fails)
    console.print(f"\n[bold {color}]selfcheck: {passed}/{len(results)} passed[/]")
    return 1 if fails else 0
