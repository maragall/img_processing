"""Stage 2 – convert to row/col filenames, fill blank grid cells, echo MIST params."""

from __future__ import annotations
import numpy as np, pandas as pd, tifffile, os
from pathlib import Path
from stitcher_pipeline.constants import ROWCOL_TEMPLATE, DEFAULT_TILE_SHAPE
from stitcher_pipeline.utils import discover_channels
from collections import Counter

def generate_stage(tile_dir: Path) -> None:
    df = pd.read_csv(tile_dir / "coordinates.csv")

    # unique sorted coordinates
    xs, ys = np.sort(df["x (mm)"].unique()), np.sort(df["y (mm)"].unique())
    gw, gh = len(xs), len(ys)

    ch_suffixes = discover_channels(tile_dir)
    dtype = tifffile.TiffFile(next(tile_dir.glob(f"*{ch_suffixes[0]}"))).pages[0].dtype
    blank = np.zeros(DEFAULT_TILE_SHAPE, dtype=dtype)

    present = set()
    for _, row in df.iterrows():
        col = int(np.where(xs == row["x (mm)"])[0][0])
        r   = int(np.where(ys == row["y (mm)"])[0][0])
        present.add((r, col))
        _rename_files(tile_dir, row["fov"], r, col, ch_suffixes)

    missing = [(r, c) for r in range(gh) for c in range(gw) if (r, c) not in present]
    for r, c in missing:
        for suf in ch_suffixes:
            tifffile.imwrite(tile_dir / ROWCOL_TEMPLATE.format(row=r, col=c, suffix=suf), blank)

    print("[generate-params] grid =", gw, "×", gh,
          "| padded", len(missing), "cells")
    print("Paste into MIST:")
    print(f"  Filename Pattern Type : Row-Column")
    print(f"  Filename Pattern      : manual_r{{rr}}_c{{cc}}_0_{ch_suffixes[0]}")
    print("  Starting Point        : Upper Left")
    print("  Direction             : Horizontal Continuous")
    print(f"  Grid Width            : {gw}")
    print(f"  Grid Height           : {gh}")
    print("  Start Row / Col       : 0")

def _rename_files(tile_dir: Path, fov: int, r: int, c: int, ch: list[str]) -> None:
    for suf in ch:
        src = tile_dir / f"manual_{fov:03d}_0_{suf}"
        dst = tile_dir / ROWCOL_TEMPLATE.format(row=r, col=c, suffix=suf)
        if src.exists():
            os.rename(src, dst)