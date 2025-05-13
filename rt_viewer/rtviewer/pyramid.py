from typing import List, Dict
import numpy as np
import dask.array as da

from .datasource import DataSource


class PyramidBuilder:
    """
    Builds downsampled overview mosaics at specified zoom levels.

    Uses Dask's coarsen to perform block-mean downsampling on the base overview.
    """
    def __init__(self, datasource: DataSource, zoom_levels: List[int]) -> None:
        """
        Initialize with a DataSource and desired downsampling factors.

        Parameters
        ----------
        datasource : DataSource
            Source for the full-resolution overview (level=1).
        zoom_levels : List[int]
            Downsampling factors for each pyramid level (e.g. [4, 8, 16]).
        """
        self.datasource = datasource
        self.zoom_levels = zoom_levels

    def build_levels(self) -> Dict[int, da.Array]:
        """
        Generate downsampled overview arrays for each zoom level.

        Returns
        -------
        Dict[int, dask.array.Array]
            Mapping from downsampling factor to Dask array of the overview at that level.
        """
        # Load the full-resolution overview (level=1)
        base_overview = self.datasource.load_overview(level=1)

        levels: Dict[int, da.Array] = {}
        for factor in self.zoom_levels:
            if factor <= 1:
                # Level 1 or invalid factor: return base overview
                levels[factor] = base_overview
            else:
                # Block-mean downsampling
                downsampled = da.coarsen(
                    np.mean,
                    base_overview,
                    {0: factor, 1: factor},
                    trim_excess=True
                )
                levels[factor] = downsampled

        return levels
