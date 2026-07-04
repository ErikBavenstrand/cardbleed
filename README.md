# cardbleed

[![CI](https://github.com/ErikBavenstrand/cardbleed/actions/workflows/ci.yml/badge.svg)](https://github.com/ErikBavenstrand/cardbleed/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cardbleed)](https://pypi.org/project/cardbleed/)

Extend the borders of card scans outward for printing.

Scans of trading cards found online often have borders that are too thin — the
printed card ends up looking wrong once cut. `cardbleed` grows the border
uniformly on all four edges by **continuing the existing border pattern**:
holofoil speckle stays speckle, solid yellow/grey borders stay clean, and
real border gradients are continued naturally instead of smearing into
streaks.

**The original image data is never re-encoded:**

| Format | Guarantee |
| --- | --- |
| PNG → PNG | original pixels bit-identical (lossless re-serialize) |
| WebP → WebP | written lossless; decoded original pixels preserved exactly |
| JPEG → JPEG | DCT-domain surgery: original coefficient blocks copied bit-exact into a larger grid; only the new border blocks are encoded, using the file's own quantization tables |

## Install

```bash
uv tool install cardbleed        # or: pipx install cardbleed
```

One-off run without installing:

```bash
uvx cardbleed card.png --compare
```

Straight from GitHub (e.g. before a PyPI release):

```bash
uv tool install git+https://github.com/ErikBavenstrand/cardbleed
```

## Usage

```bash
cardbleed card.png --compare              # extend 16px, write a QA sheet
cardbleed ./cards/ -e 2.5mm --recursive   # batch a folder, mm-based sizing
cardbleed card.jpg -e 20 --fix-aspect     # fix aspect ratio first, then extend
cardbleed card.png --target 69x94mm       # pad to an exact final size (bleed)
```

Outputs are written next to the input (or to `--out-dir`) with an `_ext`
suffix — inputs are never overwritten. `--compare` also writes a side-by-side
QA sheet with the original boundary marked so you can hunt for the seam.

## Example

The demo card below is procedurally generated
([examples/make_demo.py](examples/make_demo.py)) — no copyrighted scans. It
mimics real scan traits: a speckled border with an inward-darkening gradient,
a 1px scanner-bloom line at the edge (auto-trimmed), and a bright inner frame
line (auto-detected so sampling never crosses it).

<table>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card.png" width="170"><br><sub>input (400×550)</sub></td>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card_smart.png" width="190"><br><sub><code>-e 24</code> (smart, default)</sub></td>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card_naive.png" width="190"><br><sub><code>-e 24 --mode naive</code></sub></td>
</tr>
<tr>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card_mirror.png" width="190"><br><sub><code>-e 24 --jitter 0 --jitter-cross 0 --noise 0</code><br>(deterministic mirror continuation)</sub></td>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card_soft.png" width="190"><br><sub><code>-e 24 --smudge 2.5 --noise 0.8</code><br>(heavy smudge + grain)</sub></td>
<td align="center"><img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_card_smart_compare.png" width="190"><br><sub><code>--compare</code> QA sheet<br>(original boundary marked)</sub></td>
</tr>
</table>

Zoomed left-edge detail (extension + original border), one panel per setting —
smart · naive · mirror · soft:

<img src="https://raw.githubusercontent.com/ErikBavenstrand/cardbleed/main/examples/demo_detail_modes.png" width="600">

Smart re-randomizes the speckle while continuing the tone gradient; naive
shows the streaking that plain edge replication produces; mirror is the fully
deterministic pattern continuation; soft trades texture for smoothness.

## Options

Run `cardbleed --help` for the full flag reference. The essentials:

- `-e/--extend` — border to add per edge, in px (`16`) or mm (`2.5mm`),
  with `--left/--right/--top/--bottom` per-edge overrides.
- `--fix-aspect` — first pad the short axis to the exact card aspect ratio
  (default `--card-size 63x88` mm), then extend.
- `--mode smart|naive` — smart continues tone + texture separately; naive
  replicates the outermost line (plus the same noise/smudge).
- `-k/--sample`, `--trim`, `--jitter`, `--jitter-smooth`, `--jitter-cross`,
  `--noise`, `--smudge` — synthesis knobs; the defaults self-tune to each
  border (measured grain, per-edge bloom trim, band clamping).
- `--seed` — output is fully deterministic per file.

## How it works

Each edge is analyzed on the original image: scanner-bloom lines are detected
and trimmed, and the sampling band is clamped before inner border structure
(frame lines, artwork). The border is then split into a smooth *tone*
(continued outward mirrored, so gradients read naturally and the seam is
continuous by construction) and a speckle *residual* (resampled with a
smoothed random depth per pixel plus along-edge wobble), topped with noise
matched to the border's own measured grain and a ramped smudge. Corners are
filled in two passes so they inherit real side texture.

For JPEG output the extension amounts align to the MCU grid (8/16 px); the
tool shifts the remainder between opposite edges so final dimensions stay
exact.

## Development

```bash
git clone https://github.com/ErikBavenstrand/cardbleed
cd cardbleed
uv run cardbleed --selfcheck            # built-in assertion suite
uv run cardbleed --selfcheck scan.png   # + checks against a real scan
```

## License

[MIT](LICENSE)
