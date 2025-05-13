import threading
from typing import Tuple
import numpy as np
from cachetools import LRUCache
from .datasource import DataSource


class TileCache:
    """
    LRU cache for image tiles, evicting based on total memory usage in bytes.

    Thread-safe: uses a threading.Lock around cache operations.
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
        self.max_bytes = max_bytes
        # Cache with byte-based eviction
        self.cache: LRUCache[Tuple[int, int, int], np.ndarray] = LRUCache(
            maxsize=max_bytes,
            getsizeof=lambda arr: arr.nbytes
        )
        self._lock = threading.Lock()

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

        Raises
        ------
        TypeError
            If the loaded tile is not a NumPy ndarray.
        ValueError
            If tile size exceeds cache maximum.
        Exception
            Propagated from DataSource.load_tile if I/O fails.
        """
        key = (int(fov), int(z), int(level))
        with self._lock:
            if key in self.cache:
                return self.cache[key]

            # Load under lock to prevent duplicate loads
            tile = self.datasource.load_tile(fov, z, level)

            if not isinstance(tile, np.ndarray):
                raise TypeError(
                    "Tile must be a numpy.ndarray, got {type(tile).__name__}"
                )
            size = tile.nbytes
            if size > self.max_bytes:
                raise ValueError(
                    f"Tile size {size} bytes exceeds cache maximum of {self.max_bytes} bytes"
                )

            self.cache[key] = tile
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

        Raises
        ------
        TypeError
            If tile is not a NumPy ndarray.
        ValueError
            If tile.nbytes exceeds max_bytes.
        """
        if not isinstance(tile, np.ndarray):
            raise TypeError(
                "Tile must be a numpy.ndarray, got {type(tile).__name__}"
            )
        size = tile.nbytes
        if size > self.max_bytes:
            raise ValueError(
                f"Tile size {size} bytes exceeds cache maximum of {self.max_bytes} bytes"
            )
        with self._lock:
            self.cache[(int(fov), int(z), int(level))] = tile
