# registration/update_coordinates.py

from pathlib import Path
import re
import numpy as np
import pandas as pd
import logging
import sys

logger = logging.getLogger(__name__)

def update_coordinates(tile_dir: Path) -> None:
    """
    After MIST has written Fluo*_global-positions-*.txt, 
    recalibrate your 0/coordinates.csv → coordinates_calibrated.csv.
    """
    # 1) Load original coords
    coords_csv = tile_dir / "coordinates.csv"
    if not coords_csv.is_file():
        logger.error(f"Coordinates file not found: {coords_csv}")
        sys.exit(f"Coordinates file not found: {coords_csv}")
    df_coords = pd.read_csv(coords_csv)
    N = len(df_coords)

    # 2) Find the MIST metadata file
    parent = tile_dir.parent
    txt_files = list(parent.glob("Fluo*_global-positions-*.txt"))
    if not txt_files:
        logger.error(f"No global-positions.txt found in {parent}")
        sys.exit(f"No global-positions.txt found in {parent}")
    txt_path = txt_files[0]

    # 3) Parse pixel positions
    pattern = re.compile(
        r"manual_r(?P<r>\d+)_c(?P<c>\d+)_0_.*?position:\s*\((?P<x_px>\d+),\s*(?P<y_px>\d+)\)"
    )
    records = []
    with open(txt_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                records.append({k: int(v) for k, v in m.groupdict().items()})
    df_txt = pd.DataFrame(records)

    # 4) Map back to original mm-grid
    xs = np.sort(df_coords["x (mm)"].unique())
    ys = np.sort(df_coords["y (mm)"].unique())
    df_map = df_coords[["fov", "x (mm)", "y (mm)"]].copy()
    df_map["c"] = df_map["x (mm)"].apply(lambda x: int(np.where(xs == x)[0][0]))
    df_map["r"] = df_map["y (mm)"].apply(lambda y: int(np.where(ys == y)[0][0]))
    df_map.rename(
        columns={"x (mm)": "x_mm_orig", "y (mm)": "y_mm_orig"},
        inplace=True
    )
    dfm = df_map.merge(df_txt, on=["r", "c"], how="inner", validate="one_to_one")
    assert len(dfm) == N, f"Joined {len(dfm)}/{N} tiles; check mapping."

    # 5) Fit mm↔px linear model
    Sx, Bx = np.polyfit(dfm["x_px"], dfm["x_mm_orig"], 1)
    Sy, By = np.polyfit(dfm["y_px"], dfm["y_mm_orig"], 1)
    print("Fitted mapping:")
    print(f"  X: slope = {Sx:.6f} mm/px, intercept = {Bx:.6f} mm")
    print(f"  Y: slope = {Sy:.6f} mm/px, intercept = {By:.6f} mm\n")


    # 6) Apply calibration
    dfm["x_mm_cal"] = Sx * dfm["x_px"] + Bx
    dfm["y_mm_cal"] = Sy * dfm["y_px"] + By

    # 7) Merge back and save
    df_final = df_coords.merge(
        dfm[["fov", "x_mm_cal", "y_mm_cal"]],
        on="fov", validate="one_to_one"
    ).drop(columns=["x (mm)", "y (mm)"]).rename(
        columns={"x_mm_cal": "x (mm)", "y_mm_cal": "y (mm)"}
    )

    out_csv = tile_dir/ "coordinates.csv"
    df_final.to_csv(out_csv, index=False)
    print(f"Calibrated coordinates written to {out_csv}")
    print(df_final.head())
