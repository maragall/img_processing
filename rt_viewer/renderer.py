import numpy as np
import pandas as pd
from .cache import TileCache


class TileRenderer:
    """
    Renders tiles into a composite image for display using cached tile data and offsets.
    """
    def __init__(self, cache: TileCache, z_index: int = 0) -> None:
        """
        Initialize the tile renderer.

        Parameters
        ----------
        cache : TileCache
            Cache providing `get(fov, z, level)` for tile retrieval.
        z_index : int
            Z-slice index to render.
        """
        self.cache = cache
        self.z = z_index

    def composite(self, offsets: pd.DataFrame, level: int) -> np.ndarray:
        """
        Composite tiles into a single downsampled mosaic based on provided offsets.

        Parameters
        ----------
        offsets : pd.DataFrame
            DataFrame with columns ['fov', 'dx', 'dy'] containing pixel offsets at full resolution.
        level : int
            Downsampling factor (must divide tile dimensions).

        Returns
        -------
        np.ndarray
            Composite image as a 2D NumPy array (first channel) at downsampled resolution.
        """
        if offsets.empty:
            return np.zeros((1, 1), dtype=np.uint8)

        # Determine tile dimensions from first tile
        first_fov = int(offsets['fov'].iloc[0])
        tile_full = self.cache.get(first_fov, self.z, 1)
        # If multi-channel, select first channel
        arr0 = tile_full[0] if tile_full.ndim == 3 else tile_full
        H, W = arr0.shape

        # Compute downsampled tile size (ceil to cover full tile)
        Hs = int(np.ceil(H / level))
        Ws = int(np.ceil(W / level))

        # Convert full-resolution offsets to downsampled grid coordinates
        xs = (offsets['dx'] / level).astype(int)
        ys = (offsets['dy'] / level).astype(int)

        # Determine output mosaic dimensions
        min_x, min_y = xs.min(), ys.min()
        max_x, max_y = xs.max(), ys.max()
        out_h = (max_y - min_y) + Hs
        out_w = (max_x - min_x) + Ws

        # Initialize composite canvas
        composite = np.zeros((out_h, out_w), dtype=arr0.dtype)

        # Paste each downsampled tile into the canvas
        for _, row in offsets.iterrows():
            fov = int(row['fov'])
            dx_ds = int(row['dx'] / level) - min_x
            dy_ds = int(row['dy'] / level) - min_y

            tile_full = self.cache.get(fov, self.z, 1)
            arr = tile_full[0] if tile_full.ndim == 3 else tile_full
            # Simple downsample by strided sampling
            arr_ds = arr[::level, ::level]
            h_ds, w_ds = arr_ds.shape

            composite[dy_ds:dy_ds + h_ds, dx_ds:dx_ds + w_ds] = arr_ds

        return composite
