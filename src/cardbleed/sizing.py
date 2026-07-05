"""Amount/size parsing and per-edge extent resolution (px, mm, targets)."""

from __future__ import annotations

from .errors import FileError


def parse_amount(s: str) -> tuple[float, str]:
    raw, s = s, s.strip().lower()
    unit = "px"
    for u in ("mm", "px"):
        if s.endswith(u):
            unit, s = u, s[: -len(u)]
    try:
        return float(s), unit
    except ValueError:
        raise FileError(
            f"cannot parse amount {raw!r} (expected e.g. '16', '16px' or '2.5mm')"
        ) from None


def parse_size(s: str) -> tuple[float, float, str]:
    raw, s = s, s.strip().lower()
    unit = "px"
    for u in ("mm", "px"):
        if s.endswith(u):
            unit, s = u, s[: -len(u)]
    w, sep, h = s.partition("x")
    try:
        if not sep:
            raise ValueError
        return float(w), float(h), unit
    except ValueError:
        raise FileError(
            f"cannot parse size {raw!r} (expected WxH, e.g. '69x94mm' or '440x600')"
        ) from None


def resolve_extents(
    args, size: tuple[int, int], mcu: tuple[int, int] | None, notes: list[str]
) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) extension in px.

    Order of composition: optional aspect-ratio fill first, then the uniform /
    per-edge / target extension on top. For JPEG (mcu set), left/top must be
    MCU-aligned; the remainder is shifted to right/bottom so final dimensions
    (and thus aspect ratio and --target sizes) stay exact.
    """
    W0, H0 = size
    card_w, card_h, _ = parse_size(args.card_size)  # unit ignored: always mm
    ppm_x, ppm_y = W0 / card_w, H0 / card_h

    # -- aspect verification / fill ----------------------------------------
    ratio = card_w / card_h
    want_h = round(W0 / ratio)
    want_w = round(H0 * ratio)
    aL = aT = aR = aB = 0
    off = not (want_h == H0 or want_w == W0)
    if args.fix_aspect:
        if want_h > H0:  # too wide -> grow height
            pad = want_h - H0
            aT, aB = pad // 2, pad - pad // 2
            ppm_y = ppm_x  # unpadded axis is the physical reference now
        elif want_w > W0:  # too tall -> grow width
            pad = want_w - W0
            aL, aR = pad // 2, pad - pad // 2
            ppm_x = ppm_y
        if aL or aT or aR or aB:
            notes.append(
                f"aspect fill: +{aL}/+{aR} left/right, +{aT}/+{aB} "
                f"top/bottom to reach {card_w:g}:{card_h:g}"
            )
    elif off:
        notes.append(
            f"aspect ratio {W0}x{H0} deviates from "
            f"{card_w:g}:{card_h:g} (want ~{want_w}x{H0} or "
            f"{W0}x{want_h}); use --fix-aspect to pad it square"
        )

    def to_px(amount: str, ppm: float) -> int:
        v, unit = parse_amount(amount)
        px = round(v * ppm) if unit == "mm" else round(v)
        if px < 0:
            raise FileError(f"negative extension {amount!r}")
        return px

    # -- main extension ------------------------------------------------------
    if args.target:
        if any(v is not None for v in (args.left, args.right, args.top, args.bottom)):
            raise FileError(
                "--target cannot be combined with per-edge "
                "overrides (--left/--right/--top/--bottom)"
            )
        tw, th, unit = parse_size(args.target)
        tw_px = round(tw * ppm_x) if unit == "mm" else round(tw)
        th_px = round(th * ppm_y) if unit == "mm" else round(th)
        aw, ah = W0 + aL + aR, H0 + aT + aB
        if tw_px < aw or th_px < ah:
            raise FileError(
                f"--target {args.target} is smaller than "
                f"{aw}x{ah} (image incl. aspect fill)"
            )
        eL = (tw_px - aw) // 2
        eR = tw_px - aw - eL
        eT = (th_px - ah) // 2
        eB = th_px - ah - eT
    else:
        eL = to_px(args.left if args.left is not None else args.extend, ppm_x)
        eR = to_px(args.right if args.right is not None else args.extend, ppm_x)
        eT = to_px(args.top if args.top is not None else args.extend, ppm_y)
        eB = to_px(args.bottom if args.bottom is not None else args.extend, ppm_y)

    L, T, R, B = aL + eL, aT + eT, aR + eR, aB + eB

    # -- JPEG MCU alignment ---------------------------------------------------
    # the original blocks can only sit at multiples of the MCU from the
    # top-left corner, so left/top snap to the NEAREST feasible offset and
    # the difference moves to the opposite edge, keeping final dimensions
    # exact
    if mcu is not None:

        def align(a: int, b: int, m: int, lo: str, hi: str) -> tuple[int, int]:
            if a % m == 0:
                return a, b
            snapped = round(a / m) * m
            if snapped > a + b:  # opposite edge cannot absorb the difference
                snapped = (a // m) * m
            notes.append(
                f"JPEG alignment: {lo} edge {a}->{snapped}px ({lo} offset "
                f"must be a multiple of {m}; difference moved to the {hi} "
                f"edge — use multiples of {m} for symmetric extension)"
            )
            return snapped, b + (a - snapped)

        L, R = align(L, R, mcu[0], "left", "right")
        T, B = align(T, B, mcu[1], "top", "bottom")
    return L, T, R, B
