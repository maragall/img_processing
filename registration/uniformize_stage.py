"""Stage 3 – enforce one common tile shape for every TIFF."""

from __future__ import annotations
from pathlib import Path
import numpy as np, tifffile, os
from collections import Counter
from registration.utils import iter_tiffs, center_crop, zero_pad, overwrite_tiff
from registration.constants import DEFAULT_TILE_SHAPE

def uniformize_stage(tile_dir: Path, target_shape=DEFAULT_TILE_SHAPE) -> None:
    # gather stats
    shapes = {}
    for p in iter_tiffs(tile_dir, "manual_r*_c*_0_*.tiff"):
        with tifffile.TiffFile(p) as tif:
            shp = tif.pages[0].shape
        shapes[p] = shp

    counts = Counter(shapes.values())
    modal = counts.most_common(1)[0][0]
    if modal != target_shape:
        target_shape = modal  # follow majority

    fixed = 0
    for p, shp in shapes.items():
        if shp == target_shape:
            continue
        with tifffile.TiffFile(p) as tif:
            img = tif.pages[0].asarray()
        img = (center_crop(img, target_shape)
               if (shp[0] >= target_shape[0] and shp[1] >= target_shape[1])
               else zero_pad(img, target_shape))
        overwrite_tiff(p, img)
        fixed += 1

    print(f"[uniformize] target = {target_shape}, fixed {fixed} tiles")