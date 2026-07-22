"""Command-line interface (click + rich-click)."""

from __future__ import annotations

from types import SimpleNamespace

import rich_click as click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from ._version import __version__
from .errors import FileError
from .process import iter_inputs, process_file
from .selfcheck import selfcheck

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.STYLE_OPTIONS_TABLE_LEADING = 0
click.rich_click.OPTION_GROUPS = {
    "*": [
        {
            "name": "Sizing",
            "options": [
                "--extend",
                "--left",
                "--right",
                "--top",
                "--bottom",
                "--target",
                "--fix-aspect",
                "--card-size",
            ],
        },
        {
            "name": "Synthesis",
            "options": [
                "--mode",
                "--sample",
                "--trim",
                "--jitter",
                "--jitter-smooth",
                "--jitter-cross",
                "--shuffle",
                "--noise",
                "--smudge",
                "--seam-feather",
                "--corner-guard",
                "--halo",
                "--seed",
            ],
        },
        {
            "name": "Input / Output",
            "options": [
                "--out-dir",
                "--suffix",
                "--compare",
                "--force",
                "--recursive",
                "--dry-run",
                "--version",
                "--help",
            ],
        },
    ]
}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("inputs", nargs=-1, metavar="[INPUTS]...")
# sizing ---------------------------------------------------------------------
@click.option(
    "-e",
    "--extend",
    default="16",
    show_default=True,
    metavar="AMT",
    help="Border to add per edge: px ([cyan]16[/]) or mm ([cyan]2.5mm[/]).",
)
@click.option(
    "--left",
    default=None,
    metavar="AMT",
    help="Override for the left edge ([cyan]0[/] skips it).",
)
@click.option(
    "--right", default=None, metavar="AMT", help="Override for the right edge."
)
@click.option("--top", default=None, metavar="AMT", help="Override for the top edge.")
@click.option(
    "--bottom", default=None, metavar="AMT", help="Override for the bottom edge."
)
@click.option(
    "--target",
    default=None,
    metavar="WxH",
    help="Pad (centered) to an exact final size, e.g. "
    "[cyan]69x94mm[/] or [cyan]440x600[/]; overrides -e.",
)
@click.option(
    "--fix-aspect",
    is_flag=True,
    help="First pad the short axis so the image matches the card "
    "aspect ratio exactly, then add the border.",
)
@click.option(
    "--card-size",
    default="63x88",
    show_default=True,
    metavar="WxH",
    help="Physical card size in mm — basis for all mm math "
    "(embedded file DPI is never trusted).",
)
# synthesis ------------------------------------------------------------------
@click.option(
    "--mode",
    type=click.Choice(["smart", "pattern", "naive"]),
    default="pattern",
    show_default=True,
    help="[cyan]smart[/]: stochastic band resampling · "
    "[cyan]pattern[/]: structure-preserving randomized-mirror "
    "continuation (auto-detects repeating patterns and keeps "
    "them phase-aligned) · [cyan]naive[/]: replicate the "
    "outermost clean line.",
)
@click.option(
    "-k",
    "--sample",
    type=int,
    default=12,
    show_default=True,
    metavar="N",
    help="How many border pixels to sample patterns/colors from "
    "(auto-clamped before inner border structure).",
)
@click.option(
    "--trim",
    default="auto",
    show_default=True,
    metavar="auto|N",
    help="Outermost pixels treated as scanner bloom / cut-line junk: "
    "excluded from sampling and (png/webp) replaced. "
    "[cyan]auto[/] detects hard bloom lines per edge (max 3).",
)
@click.option(
    "--jitter",
    type=float,
    default=0.85,
    show_default=True,
    help="0..1 randomness of the sampling depth "
    "([cyan]0[/] = plain pattern continuation).",
)
@click.option(
    "--jitter-smooth",
    type=float,
    default=1.2,
    show_default=True,
    metavar="SIGMA",
    help="Smoothing of the jitter field; matches speckle grain size "
    "([cyan]0[/] = per-pixel salt & pepper).",
)
@click.option(
    "--jitter-cross",
    type=float,
    default=4.0,
    show_default=True,
    metavar="PX",
    help="Local along-edge wobble of the sampling position; kills "
    "repeated-fleck trails ([cyan]0[/] = perfectly straight).",
)
@click.option(
    "--shuffle",
    type=float,
    default=48.0,
    show_default=True,
    metavar="PX",
    help="Long-range texture borrowing along the edge (smoothed "
    "patches): holo speckle is drawn from elsewhere on the "
    "border so patterns never near-repeat in either direction "
    "([cyan]0[/] = only local wobble).",
)
@click.option(
    "--noise",
    type=float,
    default=0.35,
    show_default=True,
    metavar="F",
    help="Added grain as a multiple of the border's own measured "
    "grain (self-tuning; [cyan]0[/] = off).",
)
@click.option(
    "--smudge",
    type=float,
    default=0.6,
    show_default=True,
    metavar="SIGMA",
    help="Gaussian smudge of the new border, ramped toward the "
    "outer edge ([cyan]0[/] = off).",
)
@click.option(
    "--seam-feather",
    type=int,
    default=3,
    show_default=True,
    metavar="PX",
    help="Pixels over which randomness ramps in from the seam.",
)
@click.option(
    "--corner-guard",
    type=int,
    default=12,
    show_default=True,
    metavar="PX",
    help="Keep sampling this far away from image corners (avoids "
    "seeding from rounded/white scan corners).",
)
@click.option(
    "--edge-fill",
    type=click.Choice(["auto", "off"]),
    default="auto",
    show_default=True,
    help="[cyan]auto[/]: continue the border across transparent / "
    "rounded-corner / empty edge rows instead of extending the "
    "black gap (no-op when an edge has none). [cyan]off[/]: disable.",
)
@click.option(
    "--halo",
    type=click.Choice(["auto", "overwrite", "blend"]),
    default="auto",
    show_default=True,
    help="Trimmed halo ring handling: [cyan]overwrite[/] it "
    "(png/webp default) or [cyan]blend[/] it out (jpeg "
    "default; overwrite on jpeg re-encodes the outer "
    "block ring).",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="RNG seed (per-file streams derived from filename).",
)
# input/output ---------------------------------------------------------------
@click.option(
    "-o",
    "--out-dir",
    default=None,
    metavar="DIR",
    help="Output directory (default: alongside each input).",
)
@click.option(
    "--suffix",
    default="_ext",
    show_default=True,
    help="Appended to the output file stem.",
)
@click.option(
    "--compare",
    is_flag=True,
    help="Also write a QA sheet: original | result | result with "
    "the original boundary marked.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing output files (inputs are never overwritten).",
)
@click.option("--recursive", is_flag=True, help="Descend into subdirectories.")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be done without writing anything."
)
@click.option("--selfcheck", is_flag=True, hidden=True)
@click.version_option(__version__, "-V", "--version")
@click.pass_context
def cli(ctx: click.Context, **kw) -> None:
    """Extend card scan borders for printing, continuing the existing
    border pattern (holo speckle, solid colors, ...).

    Original image data is [bold]never re-encoded[/]: PNG/WebP pixels stay
    bit-identical and JPEG goes through lossless DCT-block surgery.
    INPUTS are image files and/or directories (png/jpg/jpeg/webp).

    [dim]Examples:[/]

    [dim]  cardbleed card.png --compare[/]

    [dim]  cardbleed ./cards/ -e 2.5mm --fix-aspect[/]
    """
    ctx.exit(run(SimpleNamespace(**kw)))


def run(args) -> int:
    console = Console(highlight=False)
    err = Console(stderr=True, highlight=False)
    if args.selfcheck:
        return selfcheck(args)
    if not args.inputs:
        raise click.UsageError("no inputs given (image files or directories)")

    files, input_errors = iter_inputs(list(args.inputs), args.recursive, args.suffix)
    for e in input_errors:
        err.print(f"[yellow]SKIPPED[/] {e}")
    if not files:
        err.print("[bold red]error:[/] no supported images found")
        return 2

    ok, failed = 0, len(input_errors)
    claimed: dict = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
        disable=len(files) < 3 or args.dry_run,
    ) as progress:
        task = progress.add_task("extending", total=len(files))
        for f in files:
            progress.update(task, description=f.name)
            try:
                process_file(f, args, console=console, claimed=claimed)
                ok += 1
            except FileError as e:
                err.print(f"[bold cyan]{f.name}[/]: [yellow]SKIPPED[/] — {e}")
                failed += 1
            except Exception as e:  # unexpected: report, keep the batch alive
                err.print(
                    f"[bold cyan]{f.name}[/]: [bold red]ERROR[/] — "
                    f"{type(e).__name__}: {e}"
                )
                failed += 1
            progress.advance(task)
    if len(files) > 1:
        parts = [f"[green]{ok} ok[/]"]
        if failed:
            parts.append(f"[red]{failed} failed[/]")
        console.print("[bold]done:[/] " + ", ".join(parts))
    return 1 if failed else 0
