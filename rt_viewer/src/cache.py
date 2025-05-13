from typing import Tuple
import numpy as np
from cachetools import LRUCache
from .datasource import DataSource


class TileCache:
    """
    LRU cache for image tiles, evicting based on total memory usage in bytes.
    """
    def __init__(self, datasource: DataSource, max_bytes: int) -> None:
        """
        Initialize the tile cache.

        Parameters
        ----------
        datasource : DataSource
            Underlying data source for loading tiles when missing in cache.
        max_bytes : int
            Maximum total bytes to store in cache before evicting least-recently-used tiles.
        """
        self.datasource = datasource
        # maxsize is total bytes, getsizeof returns size of each ndarray
        self.cache: LRUCache[Tuple[int, int, int], np.ndarray] = \
            LRUCache(maxsize=max_bytes, getsizeof=lambda arr: arr.nbytes)

    def get(self, fov: int, z: int, level: int) -> np.ndarray:
        """
        Retrieve a tile from cache, or load from DataSource if missing.

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
            The requested tile image.
        """
        key = (fov, z, level)
        try:
            return self.cache[key]
        except KeyError:
            tile = self.datasource.load_tile(fov, z, level)
            self.put(fov, z, level, tile)
            return tile

    def put(self, fov: int, z: int, level: int, tile: np.ndarray) -> None:
        """
        Store a tile in the cache.

        Parameters
        ----------
        fov : int
            Field-of-view identifier.
        z : int
            Z-slice index.
        level : int
            Pyramid level.
        tile : np.ndarray
            Image data to cache.
        """
        key = (fov, z, level)
        self.cache[key] = tile
