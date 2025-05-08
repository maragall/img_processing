#!/usr/bin/env python3
import re
import pandas as pd
from collections import deque

# ─── 1. File-paths ───────────────────────────────────────────────────────────
coord_path   = "/home/cephla/Downloads/10x_mouse_brain_2025-04-23_00-53-11.236590/0/coordinates.csv"
relpos_path  = "/home/cephla/Downloads/10x_mouse_brain_2025-04-23_00-53-11.236590/Fluo405_relative-positions-1.txt"

# ─── 2. Load & backup ────────────────────────────────────────────────────────
orig = pd.read_csv(coord_path)
orig.to_csv("coordinates.backup.csv", index=False)

# ─── 3. Compute row/col exactly as in generate_stage() ───────────────────────
xs = sorted(orig["x (mm)"].unique())
ys = sorted(orig["y (mm)"].unique())
orig["col"] = orig["x (mm)"].map({x:i for i,x in enumerate(xs)})
# MIST’s row 0 = topmost = largest Y
orig["row"] = orig["y (mm)"].map({y:i for i,y in enumerate(reversed(ys))})

# ─── 4. Parse relative-positions into directed edges ─────────────────────────
pat = re.compile(
    r"^(?P<dir>north|west),\s*"
    r"(?P<curr>[^,]+),\s*"
    r"(?P<prev>[^,]+),[^,]*,"
    r"\s*(?P<dx>-?\d+),\s*(?P<dy>-?\d+)"
)
edges = []
with open(relpos_path) as f:
    for L in f:
        m = pat.search(L)
        if not m: continue
        edges.append((
            m.group("prev").strip(),
            m.group("curr").strip(),
            int(m.group("dx")),
            int(m.group("dy"))
        ))

# ─── 5. Helpers to go filename⇄(row,col) ────────────────────────────────────
tile_re = re.compile(r"manual_r(?P<r>\d+)_c(?P<c>\d+)_0_.*\.tiff")
def rowcol(name):
    m = tile_re.search(name)
    return int(m.group("r")), int(m.group("c"))

# ─── 6. Build adjacency (bidirectional) keyed by (r,c) ──────────────────────
adj = {}
for prev_nm, curr_nm, dx, dy in edges:
    pr = rowcol(prev_nm); cr = rowcol(curr_nm)
    adj.setdefault(pr,[]).append((cr, dx,  dy))
    adj.setdefault(cr,[]).append((pr, -dx, -dy))

# ─── 7. BFS to assign every (r,c) a (x_pix,y_pix) ───────────────────────────
pixpos = {(0,0):(0,0)}          # anchor top-left at (0,0)
queue  = deque([(0,0)])
while queue:
    node = queue.popleft()
    x0,y0 = pixpos[node]
    for neigh, dx, dy in adj.get(node, []):
        if neigh in pixpos: continue
        pixpos[neigh] = (x0 + dx, y0 + dy)
        queue.append(neigh)

pixdf = pd.DataFrame([
    {"row":r, "col":c, "x_pix":xp, "y_pix":yp}
    for (r,c),(xp,yp) in pixpos.items()
])

# ─── 8. Merge & sanity-check ─────────────────────────────────────────────────
df = orig.merge(pixdf, on=["row","col"], how="left")
if df[["x_pix","y_pix"]].isna().any().any():
    bad = df[df.x_pix.isna()]
    raise RuntimeError(
        "Missing pixel data for:\n"
        + bad[["fov","row","col"]].to_string(index=False)
    )

# ─── 9. Convert to mm & reanchor to real coords ──────────────────────────────
pixel_size_um = 0.752           # your µm/px
mm_per_px     = pixel_size_um/1000.0

ref = df.iloc[0]                # first real tile
x0_mm = ref["x (mm)"] - ref.x_pix*mm_per_px
y0_mm = ref["y (mm)"] - ref.y_pix*mm_per_px

df["x (mm)"] = df.x_pix*mm_per_px + x0_mm
df["y (mm)"] = df.y_pix*mm_per_px + y0_mm

# ─── 10. Write out ────────────────────────────────────────────────────────────
out = df.drop(columns=["row","col","x_pix","y_pix"])
out.to_csv("coordinates.updated.csv", index=False)
print("Done → coordinates.updated.csv (backup at coordinates.backup.csv)")
