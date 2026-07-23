"""Reshape card scans to a target trim size with correct borders, for printing.

Continues the existing border pattern (holo speckle, solid colors, ...) into
any added area. On the extend path the original pixels are never re-encoded:

  PNG  -> PNG   original pixels bit-identical (lossless re-serialize)
  WebP -> WebP  written lossless; decoded original pixels preserved exactly
  JPEG -> JPEG  DCT-domain surgery: original coefficient blocks copied bit-exact

(``stretch=True`` and shaving an over-target border resample the art and are
opt-in.)

Python API::

    from cardbleed import bleed_card, Edges

    bleed_card(
        "card.png", "out.png",
        card_size=(63, 88),
        border_target=Edges.symmetric(vertical="3.92%", horizontal="5%"),
        border_current=Edges(top="2.5%", right="3%", bottom="2.4%", left="3.3%"),
        stretch=True,
        bleed="2.5mm",
    )
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ._version import __version__
from .cli import cli
from .errors import FileError
from .geometry import EDGES, Amount, Edges
from .synthesis import Params, extend_image

__all__ = [
    "Amount",
    "Edges",
    "FileError",
    "Params",
    "__version__",
    "bleed_card",
    "cli",
    "extend_image",
]

# programmatic defaults (the CLI carries its own via click)
_DEFAULTS: dict[str, Any] = {
    "mode": "pattern",
    "seed": 0,
    "noise": 0.35,
    "smudge": 0.6,
    "edge_fill": "auto",
    "fill_corners": False,
    "halo": "auto",
    "jitter": 0.85,
    "shuffle": 48.0,
    "sample": 12,
    "trim": "auto",
    "seam_feather": 3,
    "suffix": "_ext",
    "compare": False,
    "force": True,
    "recursive": False,
    "dry_run": False,
    "selfcheck": False,
}


def _spec(v: str | Edges | None) -> tuple[str | None, dict[str, str | None]]:
    """(base, per-edge overrides) from None | a uniform string | an Edges."""
    if v is None:
        return None, dict.fromkeys(EDGES)
    if isinstance(v, Edges):
        return None, {e: str(getattr(v, e)) for e in EDGES}
    return v, dict.fromkeys(EDGES)


def bleed_card(
    src: str | Path,
    dst: str | Path | None = None,
    *,
    card_size: tuple[float, float] | str = (63, 88),
    border_target: str | Edges | None = None,
    border_current: str | Edges | None = None,
    stretch: bool = False,
    crop: bool = True,
    bleed: str | Edges | None = None,
    **overrides: Any,
) -> Path:
    """Process ``src`` and write the result (to ``dst`` if given, else alongside
    ``src`` with the suffix). Returns the output path.

    ``card_size`` is ``(w_mm, h_mm)`` or ``"63x88"``. ``border_target`` /
    ``border_current`` / ``bleed`` accept a uniform string (``"5%"``) or an
    :class:`~cardbleed.Edges`. Extra keywords override synthesis/IO defaults
    (``mode``, ``seed``, ``noise``, ``out_dir``, ...).
    """
    from .process import process_file  # local import avoids an import cycle

    src = Path(src)
    cs = (
        card_size
        if isinstance(card_size, str)
        else f"{card_size[0]:g}x{card_size[1]:g}"
    )
    bt_base, bt = _spec(border_target)
    bc_base, bc = _spec(border_current)
    bl_base, bl = _spec(bleed)

    kw: dict[str, Any] = {**_DEFAULTS, "out_dir": None}
    kw.update(overrides)
    kw.update(
        card_size=cs,
        border_target=bt_base,
        border_current=bc_base,
        bleed=bl_base,
        stretch=stretch,
        crop=crop,
        inputs=[str(src)],
    )
    for e in EDGES:
        kw[f"border_target_{e}"] = bt[e]
        kw[f"border_current_{e}"] = bc[e]
        kw[f"bleed_{e}"] = bl[e]

    if dst is not None:
        dst = Path(dst)
        kw["out_dir"] = str(dst.parent)
        kw["suffix"] = "__cardbleed_tmp"

    produced = process_file(src, SimpleNamespace(**kw))
    if dst is not None and produced.resolve() != Path(dst).resolve():
        produced.replace(dst)
        return Path(dst)
    return produced
