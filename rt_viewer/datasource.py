import os
import re
import json
from pathlib import Path
from typing import Union, Dict, Tuple, List

import numpy as np
import pandas as pd
import dask.array as da
import tifffile

# Regex to parse filenames: manual_{fov}_0_{suffix}.tiff
FNAME_RE = re.compile(r"^manual_(?P<fov>\d+)_0_(?P<suffix>.+)\.tiff$", re.IGNORECASE)

ChannelTile = Tuple[str, Path]

class DataSource:
    """
    DataSource for rtviewer, with in-memory padding and renaming for 'manual' tiles.

    Parses acquisition parameters, indexes TIFF tiles, and provides methods to
    load individual tiles or mosaic overviews, ensuring a full rectangular grid
    by zero-padding missing positions.
    """
    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Root path '{self.root}' is not a directory")

        # Parse acquisition parameters
        json_path = self.root / "acquisition parameters.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing acquisition parameters.json at '{json_path}'")
        params = json.loads(json_path.read_text())
        try:
            self.sensor_pixel_size_um = float(params["sensor_pixel_size_um"])
        except KeyError:
            raise ValueError("'sensor_pixel_size_um' key missing in acquisition parameters.json")

        # Index manual-pattern TIFF tiles by FOV
        self.tiles_index: Dict[int, List[ChannelTile]] = {}
        for path in self.root.rglob("*.tiff"):
            m = FNAME_RE.match(path.name)
            if not m:
                continue
            fov = int(m.group('fov'))
            suffix = m.group('suffix')
            self.tiles_index.setdefault(fov, []).append((suffix, path))
        if not self.tiles_index:
            raise FileNotFoundError(f"No manual-pattern TIFF tiles found in '{self.root}'")

    def load_tile(self, fov: int, z: int, level: int) -> np.ndarray:
        if level != 1:
            raise ValueError(f"Unsupported level {level}: only level=1 is supported")

        channel_list = self.tiles_index.get(fov)
        if not channel_list:
            raise FileNotFoundError(f"No tile found for fov={fov}")

        # Sort channels by suffix
        channel_list = sorted(channel_list, key=lambda x: x[0])
        arrays = []
        for suffix, path in channel_list:
            arr = tifffile.imread(path)
            if arr.ndim not in (2, 3):
                raise ValueError(f"Unexpected dimensions {arr.shape} for '{path}'")
            # If multi-channel file, pick first plane
            if arr.ndim == 3:
                arr = arr[0]
            arrays.append(arr)
        return np.stack(arrays, axis=0)

    def load_overview(self, level: int) -> da.Array:
        """
        Load a zero-padded overview mosaic at pyramid level=1 as a Dask array.
        Missing tiles are filled with black images.
        """
        if level != 1:
            raise ValueError(f"Unsupported level {level}: only level=1 is supported")

        # Read coordinates.csv for z-plane 0
        coords_csv = self.root / "0" / "coordinates.csv"
        if not coords_csv.is_file():
            raise FileNotFoundError(f"Missing coordinates.csv at '{coords_csv}'")
        df = pd.read_csv(coords_csv)

        # Unique sorted stage positions in mm
        xs = np.sort(df["x (mm)"].unique())
        ys = np.sort(df["y (mm)"].unique())
        ncols, nrows = len(xs), len(ys)

        # Determine tile shape and dtype from a sample tile
        sample_fov, sample_list = next(iter(self.tiles_index.items()))
        sample_path = sample_list[0][1]
        sample_arr = tifffile.imread(sample_path)
        if sample_arr.ndim == 3:
            sample_arr = sample_arr[0]
        tile_h, tile_w = sample_arr.shape
        dtype = sample_arr.dtype

        # Prepare a blank tile
        blank = np.zeros((tile_h, tile_w), dtype=dtype)

        # Build an empty mosaic
        mosaic = np.zeros((nrows * tile_h, ncols * tile_w), dtype=dtype)

        # Map fov -> grid coordinates
        pos_map: Dict[int, Tuple[int,int]] = {}
        for _, row in df.iterrows():
            fov = int(row['fov'])
            x_mm, y_mm = row['x (mm)'], row['y (mm)']
            col = int(np.where(xs == x_mm)[0][0])
            row_i = int(np.where(ys == y_mm)[0][0])
            pos_map[fov] = (row_i, col)

        # Fill mosaic with actual tiles or blanks
        for r in range(nrows):
            for c in range(ncols):
                # find fov at (r,c)
                fov = next((f for f,(rr,cc) in pos_map.items() if rr==r and cc==c), None)
                if fov is not None:
                    arr = tifffile.imread(self.tiles_index[fov][0][1])
                    if arr.ndim == 3:
                        arr = arr[0]
                else:
                    arr = blank
                y0, x0 = r*tile_h, c*tile_w
                mosaic[y0:y0+tile_h, x0:x0+tile_w] = arr

        return da.from_array(mosaic, chunks=(tile_h, tile_w))

    def get_tile_centers(self) -> Dict[int, Tuple[float, float]]:
        """
        Read coordinates.csv and return mapping fov -> (y_center_px, x_center_px).
        Converts mm -> um -> pixels.
        """
        coords_csv = self.root / "0" / "coordinates.csv"
        if not coords_csv.is_file():
            raise FileNotFoundError(f"Missing coordinates.csv at '{coords_csv}'")
        df = pd.read_csv(coords_csv)
        centers: Dict[int, Tuple[float, float]] = {}
        for _, row in df.iterrows():
            fov = int(row['fov'])
            x_px = (float(row['x (mm)']) * 1000) / self.sensor_pixel_size_um
            y_px = (float(row['y (mm)']) * 1000) / self.sensor_pixel_size_um
            centers[fov] = (y_px, x_px)
        return centers
