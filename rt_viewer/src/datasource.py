from pathlib import Path
from typing import Union

import numpy as np
import dask.array as da


class DataSource:
    """
    DataSource for rtviewer.

    Initializes from a root directory containing tile data. Supports stub mode
    for testing without real data.
    """
    def __init__(self, root: Union[str, Path], stub: bool = True) -> None:
        """
        Parameters
        ----------
        root : Union[str, Path]
            Root directory of tiles.
        stub : bool
            If True, I/O operations are stubbed out (returning dummy arrays).
        """
        self.root: Path = Path(root)
        self.stub: bool = stub

    def load_tile(self, fov: int, z: int, level: int) -> np.ndarray:
        """
        Load a single tile for a given field of view, z-index, and pyramid level.

        Parameters
        ----------
        fov : int
            Field-of-view identifier.
        z : int
            Z-slice index.
        level : int
            Pyramid level.

        Returns
        -------
        np.ndarray
            Tile image data as a NumPy array.
        """
        if self.stub:
            # Return a dummy 2D array (e.g., 256x256 pixels) for testing
            return np.zeros((256, 256), dtype=np.uint8)

        # TODO: implement actual I/O using zarr or tifffile
        # Example (pseudocode):
        # path = self.root / f"fov_{fov}" / f"z_{z}" / f"level_{level}.zarr"
        # store = zarr.open(store=path, mode="r")
        # return store[:]

        raise NotImplementedError("Real I/O is not implemented")

    def load_overview(self, level: int) -> da.Array:
        """
        Load an overview (mosaic) at a given pyramid level as a Dask array.

        Parameters
        ----------
        level : int
            Pyramid level.

        Returns
        -------
        dask.array.Array
            Overview image as a Dask array.
        """
        if self.stub:
            # Return a dummy 2D Dask array (e.g., 1024x1024 pixels) for testing
            return da.zeros((1024, 1024), chunks=(256, 256), dtype=np.uint8)

        # TODO: implement actual I/O using zarr or map_tiles with dask
        # Example (pseudocode):
        # paths = sorted(self.root.glob(f"**/level_{level}.zarr"))
        # arrays = [da.from_zarr(str(p)) for p in paths]
        # return da.block(arrays)

        raise NotImplementedError("Real I/O is not implemented")
