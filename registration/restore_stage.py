from __future__ import annotations
from pathlib import Path
import os
import re
import pandas as pd


def restore_stage(tile_dir: Path) -> None:
    """
    Restore original TIFF filenames and remove padded blank tiles.

    Steps:
      1) Revert row/column names to padded-FOV names (manual_{fov:03d}_0_suffix), removing blanks.
      2) Strip zero-padding from padded-FOV names to original manual_{fov}_0_suffix.
    """
    tile_dir = Path(tile_dir)
    df = pd.read_csv(tile_dir / "coordinates.csv")
    xs = sorted(df["x (mm)"].unique())
    ys = sorted(df["y (mm)"].unique())

    # Pattern for row/col filenames
    rc_pattern = re.compile(r"^manual_r(?P<r>\d+)_c(?P<c>\d+)_0_(?P<suffix>.+\.tiff)$")
    renamed = removed = 0

    # 1) Revert row/col mapping and delete blank tiles
    for p in tile_dir.glob("manual_r*_c*_0_*.tiff"):
        m = rc_pattern.match(p.name)
        if not m:
            continue
        r, c = int(m.group("r")), int(m.group("c"))
        suffix = m.group("suffix")

        # delete if out-of-grid or no matching coordinate
        if r >= len(ys) or c >= len(xs):
            os.remove(p)
            removed += 1
            continue

        sub = df[(df["x (mm)"] == xs[c]) & (df["y (mm)"] == ys[r])]
        if sub.empty:
            os.remove(p)
            removed += 1
            continue

        fov = int(sub.iloc[0]["fov"])
        new_name = f"manual_{fov:03d}_0_{suffix}"
        os.rename(p, tile_dir / new_name)
        renamed += 1

    print(f"[update_coordinates] reverted {renamed} files; removed {removed} blanks")

    # Pattern for padded-FOV filenames
    zp_pattern = re.compile(r"^manual_(?P<fov>\d+)_0_(?P<suffix>.+\.tiff)$")
    stripped = 0

    # 2) Strip zero-padding from FOV indices, preserving the '_0_' separator
    for p in tile_dir.glob("manual_*_0_*.tiff"):
        m = zp_pattern.match(p.name)
        if not m:
            continue
        fov = int(m.group("fov"))
        suffix = m.group("suffix")
        new_name = f"manual_{fov}_0_{suffix}"
        os.rename(p, tile_dir / new_name)
        stripped += 1

    print(f"[update_coordinates] stripped zero-padding from {stripped} files")
