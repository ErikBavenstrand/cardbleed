"""Built-in assertion suite (`cardbleed --selfcheck [scan.png ...]`).

Runs geometry, format-identity, determinism, and texture-statistics checks
against synthetic fixtures — plus deeper scan-based checks when a real PNG
scan is passed as an input.
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
from .process import process_file
from .sizing import resolve_extents
from .synthesis import TRIM_CAP


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
        console.print(f"  {mark}  {name}"
                      + (f"  [dim]({detail})[/]" if detail else ""))

    tmp = Path(tempfile.mkdtemp(prefix="cardbleed_check_"))
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
        check("texture stats (residual std ratio)",
              0.5 <= rs / max(rb, 1e-6) <= 1.6,
              f"ratio {rs / max(rb, 1e-6):.2f}")
        # the mirrored tone keeps the extension's mean near the band mean
        mean_d = abs(synth.mean(axis=(0, 1)) - band.mean(axis=(0, 1))).max()
        check("texture stats (mean vs band within 8/255)", mean_d <= 8.0,
              f"{mean_d:.1f}")

        hp = synth - box_blur3(synth)

        def lag3(x):
            a, b = x[:, :-3].ravel(), x[:, 3:].ravel()
            a = a - a.mean()
            b = b - b.mean()
            den = math.sqrt(float((a * a).sum() * (b * b).sum()))
            return float((a * b).sum()) / max(den, 1e-6)

        hb = band - box_blur3(band)
        check("no streaks (lag-3 autocorr)", lag3(hp.mean(axis=2)) <=
              max(1.5 * abs(lag3(hb.mean(axis=2))), 0.35),
              f"synth {lag3(hp.mean(axis=2)):.2f} vs band "
              f"{lag3(hb.mean(axis=2)):.2f}")

        seam_x = 16 + t
        seam = np.abs(outa[:, seam_x - 1] - outa[:, seam_x]).mean()
        adj = [np.abs(outa[:, x] - outa[:, x + 1]).mean()
               for x in range(seam_x, seam_x + K - 1)]
        check("seam step within texture", seam <= 1.5 * max(np.median(adj), 1.0),
              f"seam {seam:.1f} vs band adj {np.median(adj):.1f}")

        # I/J: JPEG fixture — coefficient identity + decoded border sanity
        import jpeglib

        from .formats import jpeg_mcu, jpeg_paste_box
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
                a0[rows.start - ov:rows.stop - ov,
                   cols.start - oh:cols.stop - oh]))
        check("JPEG interior coefficients bit-exact", okc, f"extents {ext_j}")
        p0 = jpeglib.read_spatial(str(jp)).spatial.astype(np.float32)
        p1 = jpeglib.read_spatial(str(o)).spatial.astype(np.float32)
        bmean0 = p0[100:-100, 2:4].mean(axis=(0, 1))   # near the tone anchor
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
