from typing import List, Dict
import numpy as np
import dask.array as da

from .datasource import DataSource


class PyramidBuilder:
    """
    Builds downsampled overview mosaics at specified zoom levels.

    Uses Dask's coarsen for block-mean downsampling. Stub-mode DataSource
    supplies dummy data when tests run without real images.
    """
    def __init__(self, datasource: DataSource, zoom_levels: List[int]) -> None:
        """
        Parameters
        ----------
        datasource : DataSource
            Source of full-resolution overview images.
        zoom_levels : List[int]
            Integer downsampling factors for each pyramid level.
        """
        self.datasource = datasource
        self.zoom_levels = zoom_levels

    def build_levels(self) -> Dict[int, da.Array]:
        """
        Generate downsampled overview arrays for each zoom level.

        Returns
        -------
        Dict[int, dask.array.Array]
            Mapping from downsampling factor to Dask array of the overview.
        """
        # Load base overview at level=1 (full resolution)
        base_overview = self.datasource.load_overview(level=1)

        levels: Dict[int, da.Array] = {}
        for factor in self.zoom_levels:
            # Block-mean downsampling stub
            downsampled = da.coarsen(
                np.mean,
                base_overview,
                {0: factor, 1: factor},
                trim_excess=True,
            )
            levels[factor] = downsampled

        return levels
