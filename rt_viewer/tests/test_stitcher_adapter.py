#!/usr/bin/env python3
"""
test_mist_adapter.py — interactive Napari-based test for MISTAdapter.

As you pan and zoom in the viewer, this script will:
1. Determine which tiles are visible
2. Align those tiles via MISTAdapter
3. Compose and display the stitched mosaic on the fly
"""

import os
from pathlib import Path
import numpy as np
import tifffile
import napari
from rtviewer.stitcher_adapter import MISTAdapter


class TileDataSource:
    """
    Simple datasource that loads tiles from disk.
    Assumes files named 'manual_r{row}_c{col}_*.tif' under a root directory.
    """

    def __init__(self, root: str, num_rows: int, num_cols: int):
        self.root = Path(root)
        self.num_rows = num_rows
        self.num_cols = num_cols
        # preload one tile to get shape
        sample = next(self.root.glob("manual_r0_c0_*.tif"))
        self.tile_shape = tifffile.imread(str(sample)).shape[:2]

    def load_tile(self, row: int, col: int) -> np.ndarray:
        path = next(self.root.glob(f"manual_r{row}_c{col}_*.tif"))
        return tifffile.imread(str(path))

    def tiles_in_view(self, viewer: napari.Viewer) -> tuple[list[np.ndarray], int, int]:
        """
        Return all tiles intersecting the current view rectangle.
        For simplicity, this example always returns all tiles.
        """
        tiles = []
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                tiles.append(self.load_tile(r, c))
        return tiles, self.num_rows, self.num_cols


def compose_mosaic(
    tiles: list[np.ndarray],
    shifts: np.ndarray,
    num_rows: int,
    num_cols: int,
    tile_shape: tuple[int, int],
) -> np.ndarray:
    """
    Compose a mosaic given raw tiles and their dx/dy shifts.

    Parameters
    ----------
    tiles : list of np.ndarray
        Flattened row-major list of tiles.
    shifts : np.ndarray, shape (N, 2)
        Array of [dx, dy] for each tile_index.
    num_rows, num_cols : grid dimensions
    tile_shape : (height, width) of each tile

    Returns
    -------
    mosaic : np.ndarray
    """
    th, tw = tile_shape
    # estimate mosaic size
    total_h = num_rows * th + int(np.ptp(shifts[:,1])) + 10
    total_w = num_cols * tw + int(np.ptp(shifts[:,0])) + 10
    mosaic = np.zeros((total_h, total_w), dtype=tiles[0].dtype)

    for idx, img in enumerate(tiles):
        r, c = divmod(idx, num_cols)
        dx, dy = shifts[idx]
        y0 = r * th + int(dy - shifts[:,1].min())
        x0 = c * tw + int(dx - shifts[:,0].min())
        mosaic[y0 : y0 + th, x0 : x0 + tw] = img

    return mosaic


def main():
    # Configure paths and grid size
    tile_dir = "/absolute/path/to/0"       # directory of tiles
    num_rows, num_cols = 8, 11            # set to your grid dimensions
    fiji_dir = os.environ.get("FIJI_DIR") # or specify directly

    # Initialize components
    ds = TileDataSource(tile_dir, num_rows, num_cols)
    adapter = MISTAdapter(tile_dir, fiji_dir)

    # Start Napari viewer
    viewer = napari.Viewer(title="MISTAdapter Live Stitch Test")
    # display raw overview (optional)
    overview = np.block([
        [ds.load_tile(r, c) for c in range(num_cols)]
        for r in range(num_rows)
    ])
    viewer.add_image(overview, name="raw_overview", blending="additive", opacity=0.3)

    # placeholder for stitched layer
    stitched_layer = viewer.add_image(
        np.zeros((1,1)), name="stitched", blending="opaque"
    )

    def update_stitch(event=None):
        # 1. load visible tiles (here: all)
        tiles, rows, cols = ds.tiles_in_view(viewer)
        # 2. align via MISTAdapter
        df = adapter.align_tiles(tiles, rows, cols)
        shifts = df.sort_values("tile_index")[["dx", "dy"]].to_numpy()
        # 3. compose mosaic
        mosaic = compose_mosaic(tiles, shifts, rows, cols, ds.tile_shape)
        # 4. update layer
        stitched_layer.data = mosaic
        stitched_layer.contrast_limits = (mosaic.min(), mosaic.max())

    # hook into pan/zoom events
    viewer.camera.events.center.connect(update_stitch)
    viewer.camera.events.zoom.connect(update_stitch)

    # initial stitch
    update_stitch()

    napari.run()


if __name__ == "__main__":
    main()
