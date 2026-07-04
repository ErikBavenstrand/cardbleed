# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10", "numpy>=1.26"]
# ///
"""Generate the procedurally-made demo card used in the README.

Everything is synthetic — no copyrighted card art or scans. The card mimics
the traits cardbleed handles: a speckled border with a real inward-darkening
gradient, a 1px scanner-bloom line at the edge, and a bright inner frame line.
"""

from pathlib import Path

import numpy as np
from PIL import Image

W, H, BORDER = 400, 550, 14
rng = np.random.default_rng(42)

# border: teal base with an inward-darkening gradient
card = np.zeros((H, W, 3), np.float32)
yy, xx = np.mgrid[0:H, 0:W]
depth = np.minimum.reduce([xx, yy, W - 1 - xx, H - 1 - yy]).astype(np.float32)
base = np.array([70, 140, 150], np.float32)
card[:] = base + (28 - np.minimum(depth, BORDER) * 2.0)[..., None]

# holo-ish speckle: random bright flecks of varying size and hue
border_mask = depth < BORDER
for _ in range(2600):
    y, x = rng.integers(0, H), rng.integers(0, W)
    if not border_mask[y, x]:
        continue
    r = int(rng.integers(1, 3))
    color = rng.uniform(120, 255, 3)
    card[max(0, y - r):y + r, max(0, x - r):x + r] += color * rng.uniform(0.3, 0.9)

# bright inner frame line, then a plain "art" area with simple shapes
inner = depth >= BORDER
card[(depth >= BORDER) & (depth < BORDER + 2)] = (215, 220, 225)
art = depth >= BORDER + 2
card[art] = np.array([235, 230, 215]) - (yy[art] / H * 40)[..., None]
cy, cx = H * 0.42, W * 0.5
disc = (yy - cy) ** 2 + (xx - cx) ** 2 < 90 ** 2
card[disc & art] = (200, 120, 90)
ring = np.abs(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) - 120) < 6
card[ring & art] = (100, 110, 160)

# 1px scanner bloom at the very edge (cardbleed auto-trims this)
card[depth < 1] = np.clip(card[depth < 1] + 70, 0, 255)

card += rng.normal(0, 2.5, card.shape)  # mild global scan noise
out = Path(__file__).parent / "demo_card.png"
Image.fromarray(np.clip(card, 0, 255).astype(np.uint8)).save(out)
print(f"wrote {out}")
