import json
import re
import math
from pathlib import Path
from typing import Union, Dict, Tuple, List

import numpy as np
import dask.array as da
import tifffile

# Regex to parse filenames: manual_{fov}_{z}_{suffix}.tiff
FNAME_RE = re.compile(
    r"^manual_(?P<fov>\d+)_(?P<z>\d+)_(?P<suffix>.+)\.tiff$",
    re.IGNORECASE
)


ChannelTile = Tuple[str, Path]

class DataSource:
    """
    DataSource for rtviewer.

    Parses acquisitionparameters.json for pixel size, indexes tile TIFF files (including channels),
    and provides methods to load individual tiles or mosaic overviews.
    """
    def __init__(self, root: Union[str, Path]) -> None:
        """
        Initialize DataSource by scanning TIFFs and reading acquisition parameters.

        Parameters
        ----------
        root : Union[str, Path]
            Root directory of tile data. Must contain 'acquisition parameters.json'.

        Raises
        ------
        FileNotFoundError
            If root is not a directory, acquisition parameters.json is missing,
            or no TIFF tiles are found.
        ValueError
            If JSON is malformed or missing 'sensor_pixel_size_um'.
        """
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Root path '{self.root}' is not a directory")

        # Parse acquisition parameters
        json_path = self.root / "acquisition parameters.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing acquisition parameters.json at '{json_path}'")
        try:
            with open(json_path, 'r') as f:
                params = json.load(f)
            self.sensor_pixel_size_um = float(params["sensor_pixel_size_um"])
        except KeyError:
            raise ValueError("'sensor_pixel_size_um' key missing in acquisitionparameters.json")
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing 'acquisition parameters.json': {e}")

        # Index TIFF tiles by (fov, z) grouping multiple channels
        self.tiles_index: Dict[Tuple[int, int], List[ChannelTile]] = {}
        for path in self.root.rglob("*.tiff"):
            match = FNAME_RE.match(path.name)
            if not match:
                continue
            fov = int(match.group('fov'))
            z = int(match.group('z'))
            suffix = match.group('suffix')
            key = (fov, z)
            self.tiles_index.setdefault(key, []).append((suffix, path))

        if not self.tiles_index:
            raise FileNotFoundError(f"No TIFF tiles found in '{self.root}'")

    def load_tile(self, fov: int, z: int, level: int) -> np.ndarray:
        """
        Load a multi-channel tile for a given field of view, z-index, and pyramid level.

        Parameters
        ----------
        fov : int
            Field-of-view (tile) identifier.
        z : int
            Z-slice index.
        level : int
            Pyramid level (only level=1 supported).

        Returns
        -------
        np.ndarray
            Tile image data as a NumPy array with shape (C, H, W),
            where C is the number of channels.

        Raises
        ------
        ValueError
            If requested level != 1.
        FileNotFoundError
            If no tiles exist for the given fov and z.
        IOError
            If reading any TIFF file fails.
        """
        if level != 1:
            raise ValueError(f"Unsupported level {level}: only level=1 is supported")

        key = (fov, z)
        channel_list = self.tiles_index.get(key)
        if not channel_list:
            raise FileNotFoundError(f"No tile found for fov={fov}, z={z}")

        # Sort channels consistently by suffix
        channel_list = sorted(channel_list, key=lambda x: x[0])
        arrays = []
        for suffix, path in channel_list:
            if not path.is_file():
                raise FileNotFoundError(f"Missing channel file: '{path}'")
            try:
                arr = tifffile.imread(path)
            except Exception as e:
                raise IOError(f"Failed to read TIFF '{path}': {e}")
            # Ensure 2D grayscale
            if arr.ndim != 2:
                raise ValueError(f"Unexpected dimensions {arr.shape} for '{path}'")
            arrays.append(arr)

        # Stack into (C, H, W)
        return np.stack(arrays, axis=0)

    def load_overview(self, level: int) -> da.Array:
        """
        Load an overview mosaic at a given pyramid level as a Dask array.

        Parameters
        ----------
        level : int
            Pyramid level (only level=1 supported).

        Returns
        -------
        dask.array.Array
            Overview image as a Dask array, chunked by tile size.

        Raises
        ------
        ValueError
            If requested level != 1.
        """
        if level != 1:
            raise ValueError(f"Unsupported level {level}: only level=1 is supported")

        # Use first channel and first z-plane for overview
        sample_key = next(iter(self.tiles_index))
        sample_tiles = self.load_tile(sample_key[0], sample_key[1], level)
        # sample_tiles shape: (C, H, W); pick channel 0
        tile_h, tile_w = sample_tiles.shape[1:]
        dtype = sample_tiles.dtype

        # Determine grid layout at this z
        fov_z_pairs = [(fov, z) for (fov, z) in self.tiles_index if z == sample_key[1]]
        fovs = sorted(fov for (fov, _) in fov_z_pairs)
        num_tiles = len(fovs)
        cols = math.ceil(math.sqrt(num_tiles))
        rows = math.ceil(num_tiles / cols)

        mosaic = np.zeros((rows * tile_h, cols * tile_w), dtype=dtype)
        for idx, fov in enumerate(fovs):
            # Load first channel only for overview
            tile_arr = self.load_tile(fov, sample_key[1], level)[0]
            r = idx // cols
            c = idx % cols
            mosaic[r * tile_h:(r + 1) * tile_h,
                   c * tile_w:(c + 1) * tile_w] = tile_arr

        return da.from_array(mosaic, chunks=(tile_h, tile_w))
