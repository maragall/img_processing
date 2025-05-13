"""Shared helpers: I/O, regex, crop / pad helpers."""

from pathlib import Path
from typing import Iterable, Tuple, List
import re
import numpy as np
import tifffile

from stitcher_pipeline.constants import DEFAULT_TILE_SHAPE, FOV_SUFFIX_RE

# ------------------------------------------------------------------ I/O utils
def iter_tiffs(directory: Path, glob_pat: str = "manual_*_0_*.tiff") -> Iterable[Path]:
    """Yield TIFF paths in sorted order."""
    return sorted(directory.glob(glob_pat))

def discover_channels(directory: Path) -> List[str]:
    """Return sorted list of channel suffixes detected in *directory*."""
    pat = re.compile(FOV_SUFFIX_RE)
    return sorted({pat.match(p.name).group(2)            # type: ignore[arg-type]
                   for p in iter_tiffs(directory)
                   if pat.match(p.name)})

# --------------------------------------------------------- shape manipulators
def center_crop(arr: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    """Crop *arr* centrally to *target* shape."""
    r0 = (arr.shape[0] - target[0]) // 2
    c0 = (arr.shape[1] - target[1]) // 2
    return arr[r0 : r0 + target[0], c0 : c0 + target[1]]

def zero_pad(arr: np.ndarray, target: Tuple[int, int]) -> np.ndarray:
    """Pad *arr* with black pixels centrally to *target* shape."""
    out = np.zeros(target, dtype=arr.dtype)
    r0 = (target[0] - arr.shape[0]) // 2
    c0 = (target[1] - arr.shape[1]) // 2
    out[r0 : r0 + arr.shape[0], c0 : c0 + arr.shape[1]] = arr
    return out

# -------------------------------------------------------------- TIFF helpers
def overwrite_tiff(path: Path, image: np.ndarray) -> None:
    """Safely overwrite *path* with *image* (atomic replace)."""
    tmp = path.with_suffix(".tmp.tiff")
    tifffile.imwrite(tmp, image)
    tmp.replace(path)
