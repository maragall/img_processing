#!/usr/bin/env python3
"""
GPU‑accelerated tile‑registration pipeline that replicates the core MIST
phase‑correlation stage and writes an updated coordinates.csv with refined
(global) positions (mm).

Usage
-----
python gpu_mist_registration_pipeline.py --dir /path/to/input_dir

Expected folder layout
----------------------
input_dir/
├── acquisitionparameters.json   # contains "sensor_pixel_size_um"
└── 0/                            # sub‑directory with TIFF tiles & coordinates
    ├── coordinates.csv           # columns: fov,x (mm),y (mm)
    └── manual_{fov}_0_Fluorescence_405_nm_Ex.tiff

Dependencies
------------
• CuPy (GPU path) – falls back to NumPy if CuPy isn’t available
• tifffile, pandas, numpy
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np
import pandas as pd
import tifffile

# ---------- optional GPU backend ------------------------------------------------
try:
    import cupy as xp
    from cupyx.scipy.fft import fft2, ifft2  # CUDA FFT
    GPU = True
except ImportError:  # CPU fallback
    import numpy as xp  # type: ignore
    from numpy.fft import fft2, ifft2  # type: ignore
    GPU = False

# ---------- phase‑correlation helpers -------------------------------------------
EPS = 1e-8

def phase_correlation(imgA: np.ndarray, imgB: np.ndarray) -> tuple[float, float]:
    """Return (dy, dx) sub‑pixel shift that aligns B to A (positive means B→down/right)."""
    # send to backend array (gpu or cpu)
    a = xp.asarray(imgA, dtype=xp.float32)
    b = xp.asarray(imgB, dtype=xp.float32)

    HA, WA = a.shape
    HB, WB = b.shape
    H, W = HA + HB - 1, WA + WB - 1  # linear correlation size

    FA = fft2(a, s=(H, W))
    FB = fft2(b, s=(H, W))
    R = FA * xp.conj(FB)
    R /= xp.abs(R) + EPS  # cross‑power
    corr = ifft2(R)
    corr = xp.abs(corr)

    peak_idx = int(xp.argmax(corr))
    peak_y, peak_x = np.unravel_index(peak_idx, corr.shape)  # cpu indices ok

    # sub‑pixel quadratic interpolation on 3×3 neighborhood
    def quad_interp(mat: np.ndarray, axis: int, center_idx: int) -> float:
        if center_idx == 0 or center_idx == mat.shape[axis] - 1:
            return 0.0

        if axis == 0:
            # rows: scalar values
            c1 = mat[center_idx - 1, 1]
            c  = mat[center_idx    , 1]
            c2 = mat[center_idx + 1, 1]
        else:
            # cols: scalar values
            c1 = mat[1, center_idx - 1]
            c  = mat[1, center_idx    ]
            c2 = mat[1, center_idx + 1]

        denom = 2 * (c1 - 2 * c + c2)
        return float((c1 - c2) / denom) if denom != 0 else 0.0

    # gather local 3×3 window on CPU
    y0, y1 = max(0, peak_y - 1), min(corr.shape[0] - 1, peak_y + 1)
    x0, x1 = max(0, peak_x - 1), min(corr.shape[1] - 1, peak_x + 1)
    local = corr[y0:y1 + 1, x0:x1 + 1].get() if GPU else corr[y0:y1 + 1, x0:x1 + 1]

    dy_frac = quad_interp(local, 0, peak_y - y0)
    dx_frac = quad_interp(local, 1, peak_x - x0)

    refined_peak_y = peak_y + dy_frac
    refined_peak_x = peak_x + dx_frac

    dy = refined_peak_y - (HB - 1)
    dx = refined_peak_x - (WB - 1)

    # free GPU memory for big arrays
    if GPU:
        del a, b, FA, FB, R, corr
        xp.get_default_memory_pool().free_all_blocks()
    return float(dy), float(dx)

# ---------- grid discovery helpers ---------------------------------------------

def group_rows_by_y(fovs: list[int], y_coords: dict[int, float], tile_height_mm: float) -> list[list[int]]:
    """Group FOVs into rows using a y‑gap threshold = 0.5 * tile_height."""
    tol = 0.5 * tile_height_mm
    sorted_fovs = sorted(fovs, key=lambda f: y_coords[f])
    rows: list[list[int]] = []
    current_row: list[int] = []
    prev_y: float | None = None
    for fov in sorted_fovs:
        y = y_coords[fov]
        if prev_y is None or abs(y - prev_y) <= tol:
            current_row.append(fov)
        else:
            rows.append(current_row)
            current_row = [fov]
        prev_y = y
    if current_row:
        rows.append(current_row)
    # sort each row by x coordinate
    for row in rows:
        row.sort(key=lambda f: x_coords[f])
    return rows

# ---------- main pipeline -------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="GPU phase‑correlation tile registration (MIST‑like)")
    ap.add_argument("--dir", required=True, type=str, help="input directory containing acquisition parameters.json and subdir 0")
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    acq_json = os.path.join(root, "acquisition parameters.json")
    subdir = os.path.join(root, "0")
    coord_csv = os.path.join(subdir, "coordinates.csv")

    # --- load parameters & coords ---
    with open(acq_json, "r") as f:
        params = json.load(f)
    pixel_size_mm: float = params["sensor_pixel_size_um"] / 1000.0

    df = pd.read_csv(coord_csv)
    fovs = df["fov"].astype(int).tolist()
    global x_coords, y_coords  # used in helper above
    x_coords = {int(r["fov"]): float(r["x (mm)"]) for _, r in df.iterrows()}
    y_coords = {int(r["fov"]): float(r["y (mm)"]) for _, r in df.iterrows()}

    # --- load first tile to get size ---
    sample_file = os.path.join(subdir, f"manual_{fovs[0]}_0_Fluorescence_405_nm_Ex.tiff")
    sample_img = tifffile.imread(sample_file)
    H_tile, W_tile = sample_img.shape
    tile_w_mm = W_tile * pixel_size_mm
    tile_h_mm = H_tile * pixel_size_mm

    # --- load all images into dict (numpy) ---
    tiles: dict[int, np.ndarray] = {}
    for fov in fovs:
        filename = os.path.join(subdir, f"manual_{fov}_0_Fluorescence_405_nm_Ex.tiff")
        if not os.path.exists(filename):
            sys.exit(f"Missing TIFF for fov {fov}: {filename}")
        tiles[fov] = tifffile.imread(filename).astype(np.float32)

    # --- build neighbor pairs (grid assumption) ---
    rows = group_rows_by_y(fovs, y_coords, tile_h_mm)
    neighbor_pairs: list[tuple[int, int]] = []
    n_rows = len(rows)
    for r, row in enumerate(rows):
        n_cols = len(row)
        for c, fov in enumerate(row):
            # right neighbor
            if c < n_cols - 1:
                neighbor_pairs.append((fov, row[c + 1]))
            # below neighbor (pick tile closest in x)
            if r < n_rows - 1:
                below_row = rows[r + 1]
                x_curr = x_coords[fov]
                below_fov = min(below_row, key=lambda f: abs(x_coords[f] - x_curr))
                neighbor_pairs.append((fov, below_fov))

    print(f"GPU backend: {'CuPy' if GPU else 'NumPy'} | neighbor pairs: {len(neighbor_pairs)}")

    # --- compute pairwise shifts ---
    shifts: dict[tuple[int, int], tuple[float, float]] = {}
    for fovA, fovB in neighbor_pairs:
        dy, dx = phase_correlation(tiles[fovA], tiles[fovB])
        shifts[(fovA, fovB)] = (dy, dx)
        print(f"Pair ({fovA},{fovB}) shift = ({dy:.3f}px,{dx:.3f}px)")

    # --- global least‑squares alignment ---
    idx_map = {f: i for i, f in enumerate(fovs)}
    N = len(fovs)
    A_x, b_x, A_y, b_y = [], [], [], []
    for (fa, fb), (dy, dx) in shifts.items():
        i, j = idx_map[fa], idx_map[fb]
        dx_mm = dx * pixel_size_mm
        dy_mm = dy * pixel_size_mm
        stage_dx = x_coords[fb] - x_coords[fa]
        stage_dy = y_coords[fb] - y_coords[fa]
        row = [0] * N
        row[i], row[j] = -1, 1
        A_x.append(row); b_x.append(dx_mm - stage_dx)
        A_y.append(row.copy()); b_y.append(dy_mm - stage_dy)

    # anchor first tile
    anchor = idx_map[fovs[0]]
    row = [0] * N; row[anchor] = 1
    A_x.append(row); b_x.append(0.0)
    A_y.append(row.copy()); b_y.append(0.0)

    dx_corr = np.linalg.lstsq(np.array(A_x), np.array(b_x), rcond=None)[0]
    dy_corr = np.linalg.lstsq(np.array(A_y), np.array(b_y), rcond=None)[0]

    df_out = df.copy()
    df_out["x (mm)"] = [x_coords[f] + dx_corr[idx_map[f]] for f in fovs]
    df_out["y (mm)"] = [y_coords[f] + dy_corr[idx_map[f]] for f in fovs]
    out_path = os.path.join(subdir, "coordinates_refined.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Refined coordinates written to {out_path}")

if __name__ == "__main__":
    main()
