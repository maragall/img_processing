"""
Stage 2 – map tiles to row/column filenames, fill blank grid cells, and
*return* the MIST parameters needed by mist_stage.run_mist().

Public API
----------
generate_stage(tile_dir: Path) -> dict[str, int | str]
    Performs renaming/padding *in-place* and returns a dict containing:
        • filenamePattern  (row/column template for 405-nm channel)
        • gridWidth
        • gridHeight
"""

from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import tifffile

from stitcher_pipeline.constants import ROWCOL_TEMPLATE, DEFAULT_TILE_SHAPE
from stitcher_pipeline.utils import discover_channels

# ---------------------------------------------------------------------------


def _pattern_for_mist(channel_suffix: str) -> str:
    """Literal pattern string with {rr}/{cc} placeholders for MIST GUI."""
    return f"manual_r{{rr}}_c{{cc}}_0_{channel_suffix}"


def _rename_files(tile_dir: Path, fov: int, r: int, c: int, ch: list[str]) -> None:
    for suf in ch:
        src = tile_dir / f"manual_{fov:03d}_0_{suf}"
        dst = tile_dir / ROWCOL_TEMPLATE.format(row=r, col=c, suffix=suf)
        if src.exists():
            os.rename(src, dst)


# ---------------------------------------------------------------------------


def generate_stage(tile_dir: Path) -> dict[str, int | str]:
    """
    Rename to row/col form, fill blank cells with black tiles, print MIST hint.

    Returns
    -------
    dict
        Keys: filenamePattern, gridWidth, gridHeight
    """
    tile_dir = Path(tile_dir)
    df = pd.read_csv(tile_dir / "coordinates.csv")

    xs = np.sort(df["x (mm)"].unique())
    ys = np.sort(df["y (mm)"].unique())
    gw, gh = len(xs), len(ys)

    ch_suffixes = discover_channels(tile_dir)
    dtype = tifffile.TiffFile(next(tile_dir.glob(f"*{ch_suffixes[0]}"))).pages[0].dtype
    blank = np.zeros(DEFAULT_TILE_SHAPE, dtype=dtype)

    present: set[tuple[int, int]] = set()
    for _, row in df.iterrows():
        col = int(np.where(xs == row["x (mm)"])[0][0])
        r = int(np.where(ys == row["y (mm)"])[0][0])
        present.add((r, col))
        _rename_files(tile_dir, row["fov"], r, col, ch_suffixes)

    missing = [(r, c) for r in range(gh) for c in range(gw) if (r, c) not in present]
    for r, c in missing:
        for suf in ch_suffixes:
            tifffile.imwrite(
                tile_dir / ROWCOL_TEMPLATE.format(row=r, col=c, suffix=suf), blank
            )

    print(
        f"[generate-params] grid = {gw} × {gh} | padded {len(missing)} cells\n"
        "Paste into MIST:\n"
        "  Filename Pattern Type : Row-Column"
    )
    print(f"  Filename Pattern      : {_pattern_for_mist(ch_suffixes[0])}")
    print("  Starting Point        : Upper Left")
    print("  Direction             : Horizontal Continuous")
    print(f"  Grid Width            : {gw}")
    print(f"  Grid Height           : {gh}")
    print("  Start Row / Col       : 0")

    return {
        "filenamePattern": _pattern_for_mist(ch_suffixes[0]),
        "gridWidth": gw,
        "gridHeight": gh,
    }
