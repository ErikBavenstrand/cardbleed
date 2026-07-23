"""Command-line interface (click + rich-click).

Convention: every amount is a single scalar with a unit (``%`` | ``mm`` | ``px``);
every per-edge quantity is a uniform base flag plus ``-top/-right/-bottom/-left``
overrides; every toggle is a ``--x/--no-x`` boolean. Nothing takes a packed list.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

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
            "name": "Target & fit",
            "options": [
                "--card-size",
                "--border-target",
                "--border-target-top",
                "--border-target-right",
                "--border-target-bottom",
                "--border-target-left",
                "--border-current",
                "--border-current-top",
                "--border-current-right",
                "--border-current-bottom",
                "--border-current-left",
                "--stretch",
                "--crop",
            ],
        },
        {
            "name": "Cut bleed",
            "options": [
                "--bleed",
                "--bleed-top",
                "--bleed-right",
                "--bleed-bottom",
                "--bleed-left",
            ],
        },
        {
            "name": "Synthesis",
            "options": [
                "--mode",
                "--seed",
                "--noise",
                "--smudge",
                "--edge-fill",
                "--fill-corners",
                "--halo",
            ],
        },
        {
            "name": "Advanced (fine-tuning — sensible defaults, rarely needed)",
            "options": [
                "--jitter",
                "--shuffle",
                "--sample",
                "--trim",
                "--seam-feather",
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

_EDGES = ("top", "right", "bottom", "left")


def _edge_overrides(prefix: str, what: str) -> Callable:
    """Add ``--<prefix>-<edge>`` scalar overrides for the four edges."""

    def deco(f: Callable) -> Callable:
        for e in reversed(_EDGES):
            f = click.option(
                f"--{prefix}-{e}",
                default=None,
                metavar="AMT",
                help=f"{what}, {e} edge.",
            )(f)
        return f

    return deco


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("inputs", nargs=-1, metavar="[INPUTS]...")
# --- target & fit -----------------------------------------------------------
@click.option(
    "--card-size",
    default="63x88",
    show_default=True,
    metavar="WxH",
    help="Card trim size in mm — the target aspect + mm basis.",
)
@click.option(
    "--border-target",
    default=None,
    metavar="AMT",
    help="Intended border, all edges — enables fit (e.g. 5% or 3.15mm).",
)
@_edge_overrides("border-target", "Target border override")
@click.option(
    "--border-current",
    default=None,
    metavar="AMT",
    help="Where the border sits now, all edges (the marks).",
)
@_edge_overrides("border-current", "Measured border")
@click.option(
    "--stretch/--no-stretch",
    default=False,
    show_default=True,
    help="Un-distort the art so target borders land exactly (a small resample).",
)
@click.option(
    "--crop/--no-crop",
    default=True,
    show_default=True,
    help="Shave a border that is already thicker than target (never into art).",
)
# --- cut bleed --------------------------------------------------------------
@click.option(
    "--bleed",
    default=None,
    metavar="AMT",
    help="Trim margin added outside the card, all edges, e.g. [cyan]2.5mm[/].",
)
@_edge_overrides("bleed", "Bleed override")
# --- synthesis --------------------------------------------------------------
@click.option(
    "--mode",
    type=click.Choice(["pattern", "smart", "naive"]),
    default="pattern",
    show_default=True,
    help="How added border pixels are generated. [cyan]pattern[/]: "
    "structure-preserving continuation · [cyan]smart[/]: stochastic "
    "resampling · [cyan]naive[/]: replicate the outer line.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="RNG seed (per-file streams derived from filename).",
)
@click.option(
    "--noise",
    type=float,
    default=0.35,
    show_default=True,
    metavar="F",
    help="Grain added, as a multiple of the border's own ([cyan]0[/] = off).",
)
@click.option(
    "--smudge",
    type=float,
    default=0.6,
    show_default=True,
    metavar="SIGMA",
    help="Gaussian smudge ramped toward the outer edge ([cyan]0[/] = off).",
)
@click.option(
    "--edge-fill",
    type=click.Choice(["auto", "off"]),
    default="auto",
    show_default=True,
    help="Continue the border across transparent/empty edge rows.",
)
@click.option(
    "--fill-corners",
    is_flag=True,
    default=False,
    help="Square rounded/ragged corners into the border before bleeding "
    "(fills background pixels; artwork untouched; png/webp only).",
)
@click.option(
    "--halo",
    type=click.Choice(["auto", "overwrite", "blend"]),
    default="auto",
    show_default=True,
    help="Trimmed-halo handling (jpeg re-encodes the outer ring on overwrite).",
)
# --- advanced ---------------------------------------------------------------
@click.option(
    "--jitter",
    type=float,
    default=0.85,
    show_default=True,
    help="0..1 randomness of sampling depth ([cyan]0[/] = plain continuation).",
)
@click.option(
    "--shuffle",
    type=float,
    default=48.0,
    show_default=True,
    metavar="PX",
    help="Long-range along-edge texture borrowing.",
)
@click.option(
    "-k",
    "--sample",
    type=int,
    default=12,
    show_default=True,
    metavar="N",
    help="Border pixels sampled from (auto-clamped before inner structure).",
)
@click.option(
    "--trim",
    default="auto",
    show_default=True,
    metavar="auto|N",
    help="Outermost bloom/junk pixels excluded from sampling.",
)
@click.option(
    "--seam-feather",
    type=int,
    default=3,
    show_default=True,
    metavar="PX",
    help="Pixels over which randomness ramps in from the seam.",
)
# --- input / output ---------------------------------------------------------
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
    help="Also write a QA sheet: original | result | result w/ boundary marked.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing outputs (inputs are never overwritten).",
)
@click.option("--recursive", is_flag=True, help="Descend into subdirectories.")
@click.option("--dry-run", is_flag=True, help="Show what would be done; write nothing.")
@click.option("--selfcheck", is_flag=True, hidden=True)
@click.version_option(__version__, "-V", "--version")
@click.pass_context
def cli(ctx: click.Context, **kw: Any) -> None:
    """Reshape a card scan to a target trim size with correct borders, continuing
    the existing border pattern (holo speckle, solid colors, ...).

    Original image data is [bold]never re-encoded[/] on the extend path: PNG/WebP
    pixels stay bit-identical, JPEG uses lossless DCT-block surgery. ([cyan]--stretch[/]
    and shaving an over-target border do resample and are opt-in.)
    INPUTS are image files and/or directories (png/jpg/jpeg/webp).

    [dim]Examples:[/]

    [dim]  cardbleed card.png --bleed 2.5mm[/]

    [dim]  cardbleed card.png --card-size 63x88 --border-target 5%
          --border-target-top 3.92% --border-target-bottom 3.92%
          --border-current-top 2.5% --border-current-right 3%
          --border-current-bottom 2.4% --border-current-left 3.3% --stretch[/]
    """
    ctx.exit(run(SimpleNamespace(**kw)))


def run(args: SimpleNamespace) -> int:
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
        task = progress.add_task("processing", total=len(files))
        for f in files:
            progress.update(task, description=f.name)
            try:
                process_file(f, args, console=console, claimed=claimed)
                ok += 1
            except FileError as e:
                err.print(f"[bold cyan]{f.name}[/]: [yellow]SKIPPED[/] — {e}")
                failed += 1
            except Exception as e:  # keep the batch alive
                err.print(f"[bold cyan]{f.name}[/]: [bold red]ERROR[/] — {e!r}")
                failed += 1
            progress.advance(task)
    if len(files) > 1:
        parts = [f"[green]{ok} ok[/]"] + (
            [f"[red]{failed} failed[/]"] if failed else []
        )
        console.print("[bold]done:[/] " + ", ".join(parts))
    return 1 if failed else 0
