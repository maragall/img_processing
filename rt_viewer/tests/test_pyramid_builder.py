import numpy as np
import dask.array as da
import pytest

from rtviewer.pyramid import PyramidBuilder


class StubDataSource:
    """
    Stub DataSource returning a fixed full-resolution overview.
    """
    def __init__(self, base_array: np.ndarray):
        # Wrap base_array in a Dask array with one chunk
        self._overview = da.from_array(base_array, chunks=base_array.shape)

    def load_overview(self, level: int) -> da.Array:
        if level != 1:
            raise ValueError(f"Unsupported level {level}")
        return self._overview


def test_build_levels_identity_level1():
    # Base overview is a 4×4 array of sequential values
    base = np.arange(16, dtype=float).reshape(4, 4)
    ds = StubDataSource(base)
    # Zoom level 1 should return the base overview unchanged
    pb = PyramidBuilder(datasource=ds, zoom_levels=[1])
    levels = pb.build_levels()
    assert 1 in levels
    result = levels[1].compute()
    np.testing.assert_array_equal(result, base)


def test_build_levels_downsample_factor2():
    # Base overview is a 4×4 float array
    base = np.array([
        [ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=float)
    ds = StubDataSource(base)
    # Zoom factor 2 should produce a 2×2 downsampled overview
    pb = PyramidBuilder(datasource=ds, zoom_levels=[2])
    levels = pb.build_levels()
    assert 2 in levels
    down = levels[2].compute()
    # Expected block-means: [[3.5, 5.5], [11.5, 13.5]]
    expected = np.array([[3.5, 5.5], [11.5, 13.5]], dtype=float)
    np.testing.assert_allclose(down, expected)


def test_build_levels_downsample_factor4():
    # Base overview is a 4×4 array; factor 4 produces a 1×1 mean of all
    base = np.arange(1, 17, dtype=float).reshape(4, 4)
    ds = StubDataSource(base)
    # Zoom factor 4
    pb = PyramidBuilder(datasource=ds, zoom_levels=[4])
    levels = pb.build_levels()
    assert 4 in levels
    down = levels[4].compute()
    expected = np.array([[base.mean()]], dtype=float)
    np.testing.assert_allclose(down, expected)
