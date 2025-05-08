from pathlib import Path
import re
import numpy as np
import pandas as pd

# Parameters
MM_PER_PX = 0.000752  # mm per pixel

# Load original coordinates
coords_path = Path("/home/cephla/Downloads/10x_mouse_brain_2025-04-23_00-53-11.236590/0/coordinates.csv")
df_coords = pd.read_csv(coords_path)
N = len(df_coords)

# Parse MIST global positions
txt_path = Path("/home/cephla/Downloads/10x_mouse_brain_2025-04-23_00-53-11.236590/Fluo405_global-positions-1.txt")
pattern = re.compile(
    r"manual_r(?P<r>\d+)_c(?P<c>\d+)_0_.*?position:\s*\((?P<x_px>\d+),\s*(?P<y_px>\d+)\)"
)
records = []
with open(txt_path) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            d = {k: int(v) for k, v in m.groupdict().items()}
            records.append(d)
df_txt = pd.DataFrame(records)

# Recompute grid indices from original coords
xs = np.sort(df_coords["x (mm)"].unique())
ys = np.sort(df_coords["y (mm)"].unique())
df_map = df_coords[["fov", "x (mm)", "y (mm)"]].copy()
df_map["c"] = df_map["x (mm)"].apply(lambda x: int(np.where(xs == x)[0][0]))
df_map["r"] = df_map["y (mm)"].apply(lambda y: int(np.where(ys == y)[0][0]))
df_map.rename(columns={"x (mm)": "x_mm_orig", "y (mm)": "y_mm_orig"}, inplace=True)

# Merge on (r, c)
dfm = df_map.merge(df_txt, on=["r", "c"], how="inner", validate="one_to_one")
assert len(dfm) == N, f"Joined {len(dfm)}/{N} tiles; check mapping."

# Compute per-tile intercepts
dfm["b_x_i"] = dfm["x_mm_orig"] - MM_PER_PX * dfm["x_px"]
dfm["b_y_i"] = dfm["y_mm_orig"] - MM_PER_PX * dfm["y_px"]

# Global intercepts
b_x = dfm["b_x_i"].mean()
b_y = dfm["b_y_i"].mean()
std_x = dfm["b_x_i"].std()
std_y = dfm["b_y_i"].std()

print(f"Slope (fixed): {MM_PER_PX} mm/px")
print(f"Computed intercepts:")
print(f"  b_x = {b_x:.6f} mm (std {std_x:.6f})")
print(f"  b_y = {b_y:.6f} mm (std {std_y:.6f})\n")

# Apply calibration
dfm["x_mm_cal"] = MM_PER_PX * dfm["x_px"] + b_x
dfm["y_mm_cal"] = MM_PER_PX * dfm["y_px"] + b_y

# Build final calibrated DataFrame
df_final = df_coords.merge(
    dfm[["fov", "x_mm_cal", "y_mm_cal"]], on="fov", validate="one_to_one"
)
df_final = df_final.drop(columns=["x (mm)", "y (mm)"])
df_final = df_final.rename(columns={"x_mm_cal": "x (mm)", "y_mm_cal": "y (mm)"})

# Save to CSV
out_path = Path("/home/cephla/Downloads/10x_mouse_brain_2025-04-23_00-53-11.236590/0/coordinates_calibrated.csv")
df_final.to_csv(out_path, index=False)

# Display results
print("First 5 rows of calibrated coordinates:")
print(df_final.head())
