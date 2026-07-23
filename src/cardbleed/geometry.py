"""Sizing geometry: amounts, per-edge specs, and the border-fit solver.

One convention throughout: an :class:`Amount` is a single scalar with a unit
(``%`` | ``mm`` | ``px``); a per-edge quantity is an :class:`Edges` of four
Amounts. Nothing is ever a list packed into a string.

The fit solver reshapes a scan to an exact card aspect with target border
widths, by adding (and optionally shaving) border — never distorting the art,
unless ``stretch`` is asked for, which un-distorts it to hit the borders
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import FileError

_UNITS = ("%", "mm", "px")
EDGES = ("top", "right", "bottom", "left")


@dataclass(frozen=True, slots=True)
class Amount:
    """A scalar with a unit: ``5%``, ``2.5mm`` or ``18px``."""

    value: float
    unit: str

    def __str__(self) -> str:
        return f"{self.value:g}{self.unit}"

    @classmethod
    def parse(cls, s: str) -> Amount:
        raw, t = s, s.strip().lower()
        for u in _UNITS:
            if t.endswith(u):
                try:
                    return cls(float(t[: -len(u)]), u)
                except ValueError:
                    break
        raise FileError(
            f"cannot parse amount {raw!r} (expected e.g. '5%', '2.5mm', '18px')"
        )

    def as_px_of(self, dim_px: float, dim_mm: float) -> float:
        """Pixels, measured against one axis of an image (px + its mm size)."""
        if self.unit == "px":
            return self.value
        if self.unit == "mm":
            return self.value * dim_px / dim_mm
        return self.value / 100.0 * dim_px  # %

    def as_fraction_of_card(self, card_mm: float) -> float:
        """Fraction of a card axis (for target borders)."""
        if self.unit == "%":
            return self.value / 100.0
        if self.unit == "mm":
            return self.value / card_mm
        raise FileError("border-target in px is ambiguous; use % or mm")


@dataclass(frozen=True, slots=True)
class Edges:
    """Four per-edge :class:`Amount`\\ s (top, right, bottom, left)."""

    top: Amount
    right: Amount
    bottom: Amount
    left: Amount

    @classmethod
    def all(cls, a: Amount | str) -> Edges:
        a = a if isinstance(a, Amount) else Amount.parse(a)
        return cls(a, a, a, a)

    @classmethod
    def symmetric(cls, vertical: Amount | str, horizontal: Amount | str) -> Edges:
        """``vertical`` = top & bottom, ``horizontal`` = left & right."""
        v = vertical if isinstance(vertical, Amount) else Amount.parse(vertical)
        h = horizontal if isinstance(horizontal, Amount) else Amount.parse(horizontal)
        return cls(v, h, v, h)

    @classmethod
    def from_edges(cls, base: str | None, **overrides: str | None) -> Edges:
        """Build from a uniform base plus per-edge string overrides (the CLI
        shape). ``base`` fills any edge without an override; a fully-unset edge
        defaults to ``0px``."""
        vals: dict[str, Amount] = {}
        for e in EDGES:
            v = overrides.get(e) or base
            vals[e] = Amount.parse(v) if v is not None else Amount(0.0, "px")
        return cls(**vals)


@dataclass(frozen=True, slots=True)
class FitPlan:
    """Result of :func:`solve_fit`: how to reshape the scan."""

    stretch_x: float  # resample factors applied to the whole image (1.0 = none)
    stretch_y: float
    ext: dict[str, float]  # per-edge px to add (>0) or shave (<0), post-stretch
    trim_w: float  # final card width/height in px (exact card aspect)
    trim_h: float
    borders: dict[str, float]  # resulting border as a fraction of the card, per edge
    over_ratio: float  # how far the art aspect is off target (0 = perfect); >0 too wide
    cropped: tuple[str, ...]  # edges shaved


def _card_aspect(card_w_mm: float, card_h_mm: float) -> float:
    return card_w_mm / card_h_mm


def solve_fit(
    w: int,
    h: int,
    current: Edges,
    target: Edges,
    card_w_mm: float,
    card_h_mm: float,
    *,
    stretch: bool,
    crop: bool,
) -> FitPlan:
    """Compute the reshape that lands the outer at exactly the card aspect with
    borders as close as possible to ``target`` (exactly, if ``stretch``).

    ``current`` = where the border sits in the scan (% of image / mm / px).
    ``target``  = intended border widths (% of card / mm).
    """
    cur = {
        e: getattr(current, e).as_px_of(
            w if e in ("left", "right") else h,
            card_w_mm if e in ("left", "right") else card_h_mm,
        )
        for e in EDGES
    }
    g = {
        e: getattr(target, e).as_fraction_of_card(
            card_w_mm if e in ("left", "right") else card_h_mm
        )
        for e in EDGES
    }
    if g["left"] + g["right"] >= 1 or g["top"] + g["bottom"] >= 1:
        raise FileError("target borders sum to the whole card")

    iw = w - cur["left"] - cur["right"]
    ih = h - cur["top"] - cur["bottom"]
    if iw <= 0 or ih <= 0:
        raise FileError("border marks leave no inner frame")

    ca = _card_aspect(card_w_mm, card_h_mm)
    expected_inner = ca * (1 - g["left"] - g["right"]) / (1 - g["top"] - g["bottom"])
    over_ratio = (iw / ih) / expected_inner - 1.0  # >0: art too wide vs target

    sx = sy = 1.0
    if stretch:
        if iw / ih > expected_inner:  # too wide -> stretch height
            sy = (iw / ih) / expected_inner
        else:  # too tall -> stretch width
            sx = expected_inner / (iw / ih)
        cur = {e: cur[e] * (sx if e in ("left", "right") else sy) for e in EDGES}
        iw, ih = iw * sx, ih * sy
        w2, h2 = w * sx, h * sy
    else:
        w2, h2 = float(w), float(h)

    # closed-form scale k: final card = (card_w_mm*k) x (card_h_mm*k) px, exact
    # aspect, minimizing the border-fraction error.
    a, c = iw / card_w_mm, ih / card_h_mm
    p, q = 1 - g["left"] - g["right"], 1 - g["top"] - g["bottom"]
    u = (a * p + c * q) / (a * a + c * c)
    k_floor = (
        max(iw / card_w_mm, ih / card_h_mm)
        if crop
        else max(w2 / card_w_mm, h2 / card_h_mm)
    )
    u = min(u, 1.0 / k_floor)  # never shave into the art / never crop when off
    k = 1.0 / u
    trim_w, trim_h = card_w_mm * k, card_h_mm * k

    def split(
        budget: float, tgt_a: float, tgt_b: float, cur_a: float, cur_b: float
    ) -> tuple[float, float]:
        lo_a = -cur_a if crop else 0.0
        lo_b = -cur_b if crop else 0.0
        surplus = budget - (tgt_a - cur_a) - (tgt_b - cur_b)
        ea, eb = (tgt_a - cur_a) + surplus / 2, (tgt_b - cur_b) + surplus / 2
        if ea < lo_a:
            ea, eb = lo_a, budget - lo_a
        if eb < lo_b:
            eb, ea = lo_b, budget - lo_b
        return ea, eb

    eL, eR = split(
        trim_w - w2, g["left"] * trim_w, g["right"] * trim_w, cur["left"], cur["right"]
    )
    eT, eB = split(
        trim_h - h2, g["top"] * trim_h, g["bottom"] * trim_h, cur["top"], cur["bottom"]
    )
    ext = {"left": eL, "right": eR, "top": eT, "bottom": eB}
    borders = {
        e: (cur[e] + ext[e]) / (trim_w if e in ("left", "right") else trim_h)
        for e in EDGES
    }
    cropped = tuple(e for e in EDGES if ext[e] < -0.5)
    return FitPlan(sx, sy, ext, trim_w, trim_h, borders, over_ratio, cropped)


def resolve_bleed(
    bleed: Edges | None,
    trim_w: float,
    card_w_mm: float,
    trim_h: float,
    card_h_mm: float,
) -> dict[str, float]:
    """Per-edge cut-bleed in px, measured against the final card size."""
    if bleed is None:
        return dict.fromkeys(EDGES, 0.0)
    return {
        e: getattr(bleed, e).as_px_of(
            trim_w if e in ("left", "right") else trim_h,
            card_w_mm if e in ("left", "right") else card_h_mm,
        )
        for e in EDGES
    }


def parse_size(s: str) -> tuple[float, float, str]:
    """Parse a ``WxH`` dimension with an optional unit (``63x88mm``)."""
    raw, t = s, s.strip().lower()
    unit = "px"
    for u in ("mm", "px"):
        if t.endswith(u):
            unit, t = u, t[: -len(u)]
    w, sep, hh = t.partition("x")
    try:
        if not sep:
            raise ValueError
        return float(w), float(hh), unit
    except ValueError:
        raise FileError(
            f"cannot parse size {raw!r} (expected WxH, e.g. '63x88mm')"
        ) from None
