"""Global constants shared by stage modules."""

# Default physical tile shape (rows, cols) in pixels
DEFAULT_TILE_SHAPE = (2048, 2048)

# Row-column filename template  (two-digit zero-padded row + col)
ROWCOL_TEMPLATE = "manual_r{row:02d}_c{col:02d}_0_{suffix}"

# Regex used by utils to pull FOV / suffix out of any “manual_###_0_*.tiff”
FOV_SUFFIX_RE = r"^manual_(\d+)_0_(.+\.tiff)$"