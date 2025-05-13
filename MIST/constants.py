import os

"""Global constants shared by stage modules."""
# Default physical tile shape (rows, cols) in pixels
DEFAULT_TILE_SHAPE = (2048, 2048)

# Row-column filename template  (two-digit zero-padded row + col)
ROWCOL_TEMPLATE = "manual_r{row:02d}_c{col:02d}_0_{suffix}"

# Regex used by utils to pull FOV / suffix out of any “manual_###_0_*.tiff”
FOV_SUFFIX_RE = r"^manual_(\d+)_0_(.+\.tiff)$"

DEFAULT_MIST_PARAMS = {
    "filenamePatternType": "Row-Column",
    "startingPoint": "Upper Left",
    "direction": "Horizontal Continuous",
    "blendingMode": "Overlay",
    "displayStitchedImage": True,
    "saveFullStitchedImage": True,
    "useDoublePrecision": True,
    "translationRefinementMethod": "Single Hill Climb",
    "logLevel": "Mandatory",
    "debugLevel": "None",
    "stitchingProgram": "AUTO",
    # threads default to <=10
    "numCPUThreads": min(10, os.cpu_count() or 1),
}
