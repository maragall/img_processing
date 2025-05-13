from typing import List
import numpy as np
import pandas as pd


class MISTAdapter:
    """
    Adapter for MIST stitcher integration.

    Provides tile alignment via MIST.
    """
    def __init__(self) -> None:
        """
        Initialize MISTAdapter.
        """
        pass

    def align_tiles(self, tiles: List[np.ndarray]) -> pd.DataFrame:
        """
        Align tiles using MIST stitcher.

        Parameters
        ----------
        tiles : List[np.ndarray]
            List of image tiles to align.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['tile_index', 'dx', 'dy'], currently zero offsets.
        """
        # TODO: plug in actual MIST stitcher here
        results = []
        for idx, _tile in enumerate(tiles):
            results.append({'tile_index': idx, 'dx': 0, 'dy': 0})
        return pd.DataFrame(results)  
