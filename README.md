<h1 align="center">cardbleed</h1>

<p align="center">
  <a href="https://github.com/ErikBavenstrand/cardbleed/actions/workflows/ci.yml"><img src="https://github.com/ErikBavenstrand/cardbleed/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/cardbleed/"><img src="https://img.shields.io/pypi/v/cardbleed" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"></a>
</p>

<p align="center">
  <b>Grow the borders of card scans for printing — without touching a single original pixel.</b>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card.png" width="180">
  &nbsp;&nbsp;→&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card_smart.png" width="200">
</p>

Card scans found online often have borders that are too thin — print one and
the cut card just looks *off*. `cardbleed` extends the border on all four
edges by **continuing the pattern that's already there**:

- 🎨 **Pattern-aware** — holofoil speckle stays speckle (no streaking, no
  repeats), solid borders stay clean, and real border gradients continue
  naturally.
- 🔒 **Lossless by design** — the original image is never re-encoded; only
  border data is added (see [guarantees](#lossless-guarantees)).
- 🧠 **Self-tuning** — scanner bloom, usable border depth, and grain level
  are measured per edge, per file. The defaults just work.
- 📐 **Print-ready** — mm-based sizing, exact aspect-ratio fixing, exact
  target sizes for bleed workflows, correct DPI stamped in the output.

## Install

```bash
uv tool install cardbleed        # or: pipx install cardbleed
```

```bash
uvx cardbleed card.png           # one-off, no install
```

<sub>Before a PyPI release (or for the latest main): `uv tool install git+https://github.com/ErikBavenstrand/cardbleed`</sub>

## Quick start

```bash
cardbleed card.png --compare              # extend 16px, write a QA sheet
cardbleed ./cards/ -e 2.5mm --recursive   # batch a folder, mm-based sizing
cardbleed card.jpg -e 20 --fix-aspect     # snap to 63x88 ratio, then extend
cardbleed card.png --target 69x94mm       # pad to an exact final size (bleed)
```

Outputs land next to the input (or in `--out-dir`) with an `_ext` suffix —
**inputs are never overwritten**. `--compare` adds a side-by-side QA sheet
with the original boundary marked in magenta so you can hunt for the seam.

## Gallery

The demo card is procedurally generated
([`examples/make_demo.py`](examples/make_demo.py)) — no copyrighted scans.
It mimics real scan traits: speckled border, inward-darkening gradient, a 1px
scanner-bloom line (auto-trimmed), and a bright inner frame line (auto-detected
so sampling never crosses it).

<table>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card.png" width="170"><br><sub><b>input</b> (400×550)</sub></td>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card_smart.png" width="190"><br><sub><b>smart</b> (default)<br><code>-e 24</code></sub></td>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card_naive.png" width="190"><br><sub><b>naive</b><br><code>-e 24 --mode naive</code></sub></td>
</tr>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card_mirror.png" width="190"><br><sub><b>mirror</b> (deterministic)<br><code>--jitter 0 --jitter-cross 0 --shuffle 0 --noise 0</code></sub></td>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card_soft.png" width="190"><br><sub><b>soft</b><br><code>--smudge 2.5 --noise 0.8</code></sub></td>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card_smart_compare.png" width="190"><br><sub><b>QA sheet</b><br><code>--compare</code></sub></td>
</tr>
</table>

Zoomed left-edge detail (new border + original border) — **smart · naive ·
mirror · soft**:

<p align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_detail_modes.png" width="620"></p>

Smart re-randomizes the speckle in *both* directions (long-range `--shuffle`
borrows texture from elsewhere on the edge, so flecks neither streak outward
nor near-repeat) while continuing the tone gradient. Naive shows the streaking
that plain edge replication produces. Mirror is the fully deterministic
pattern continuation. Soft trades texture for smoothness.

## Lossless guarantees

| Format | What happens to the original data |
| --- | --- |
| **PNG → PNG** | pixels bit-identical (lossless re-serialize) |
| **WebP → WebP** | written lossless (`exact`); decoded original pixels preserved exactly |
| **JPEG → JPEG** | DCT-domain surgery: original quantized coefficient blocks copied **bit-exact** into a larger grid; only new border blocks are encoded, with the file's own quantization tables |

For JPEG, extension amounts align to the MCU grid (8/16 px) — the remainder
is shifted between opposite edges so final dimensions stay exact.

## Options

`cardbleed --help` shows the full, grouped reference. The ones you'll touch:

| Flag | Default | What it does |
| --- | --- | --- |
| `-e, --extend` | `16` | Border per edge, px or mm (`2.5mm`). Per-edge: `--left/--right/--top/--bottom` |
| `--fix-aspect` | off | Pad the short axis to the exact card ratio (`--card-size`, default `63x88` mm) first |
| `--target` | — | Pad to an exact final size instead, e.g. `69x94mm` |
| `--mode` | `smart` | `smart` (tone + texture synthesis) or `naive` (replicate + noise + smudge) |
| `-k, --sample` | `8` | Border pixels to sample patterns from (auto-clamped) |
| `--trim` | `auto` | Scanner-bloom lines to cut per edge (auto-detected, max 3) |
| `--jitter` / `--shuffle` | `0.85` / `48` | Depth randomness / long-range texture borrowing along the edge |
| `--noise` / `--smudge` | `0.35` / `0.6` | Grain (matched to the border's own) / ramped blur |
| `--seed` | `0` | Fully deterministic output per file |

<details>
<summary><b>How it works</b></summary>

Each edge is analyzed on the original image: scanner-bloom lines are detected
and trimmed, and the sampling band is clamped before inner border structure
(frame lines, artwork). The border is then split into a smooth **tone**
(continued outward mirrored, so gradients read naturally and the seam is
continuous by construction) and a speckle **residual**, resampled per pixel
with three smoothed random fields: depth jitter, a small along-edge wobble,
and a long-range patch shuffle that borrows texture from elsewhere on the
edge. Noise matched to the border's measured grain and a ramped smudge finish
the synthesis. Corners are filled in two passes so they inherit real side
texture. All randomness ramps in from zero at the seam — the first synthesized
line is an exact continuation of the edge.

</details>

## Development

```bash
git clone https://github.com/ErikBavenstrand/cardbleed && cd cardbleed
uv run cardbleed --selfcheck            # built-in assertion suite
uv run cardbleed --selfcheck scan.png   # + deeper checks against a real scan
```

The package is laid out for extensibility: `synthesis.py` (border strategies),
`formats.py` (format-preserving I/O incl. JPEG DCT surgery), `sizing.py`
(px/mm/target/aspect math), `process.py` (pipeline), `cli.py`, `selfcheck.py`.

## License

[MIT](LICENSE)
