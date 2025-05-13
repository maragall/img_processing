import numpy as np
import pandas as pd
import pytest

from rtviewer.renderer import TileRenderer


class FakeCache:
    """
    Simple cache stub returning predefined tile arrays by FOV.
    """
    def __init__(self, data):
        # data: dict of fov -> 2D or 3D NumPy array
        self.data = data

    def get(self, fov, z, level):
        return self.data[fov]


def test_composite_two_tiles_horizontal():
    # Create two 4×4 tiles with distinct constant values
    arr0 = np.zeros((4, 4), dtype=np.uint8)
    arr1 = np.full((4, 4), 2, dtype=np.uint8)

    cache = FakeCache({0: arr0, 1: arr1})
    renderer = TileRenderer(cache, z_index=0)

    # Offsets: place tile 0 at (0,0), tile 1 at (4,0)
    offsets = pd.DataFrame({'fov': [0, 1], 'dx': [0, 4], 'dy': [0, 0]})
    composite = renderer.composite(offsets, level=1)

    # Expect a 4×8 mosaic: left half zeros, right half twos
    assert composite.shape == (4, 8)
    assert np.all(composite[:, :4] == 0)
    assert np.all(composite[:, 4:] == 2)


def test_composite_downsampled():
    # Create two 4×4 tiles with increasing values
    arr0 = np.arange(16, dtype=np.uint8).reshape((4, 4))
    arr1 = (np.arange(16, dtype=np.uint8) + 100).reshape((4, 4))

    cache = FakeCache({0: arr0, 1: arr1})
    renderer = TileRenderer(cache, z_index=0)

    offsets = pd.DataFrame({'fov': [0, 1], 'dx': [0, 4], 'dy': [0, 0]})
    composite = renderer.composite(offsets, level=2)

    # Downsample by striding: arr[::2, ::2]
    expected0 = arr0[::2, ::2]
    expected1 = arr1[::2, ::2]

    # Mosaic dims: height=2, width=4
    assert composite.shape == (2, 4)
    # Left block matches expected0, right block matches expected1
    assert np.array_equal(composite[:, :2], expected0)
    assert np.array_equal(composite[:, 2:], expected1)


def test_composite_with_negative_offsets():
    # Test shifting with negative dx/dy
    arr0 = np.ones((2, 2), dtype=np.uint8) * 5
    arr1 = np.ones((2, 2), dtype=np.uint8) * 8

    cache = FakeCache({0: arr0, 1: arr1})
    renderer = TileRenderer(cache, z_index=0)

    # Offsets: tile 0 at (2,2), tile 1 at (0,0)
    offsets = pd.DataFrame({'fov': [0, 1], 'dx': [2, 0], 'dy': [2, 0]})
    comp = renderer.composite(offsets, level=1)

    # min_x=0, min_y=0, max_x=2, max_y=2; out_h=2+2=4, out_w=2+2=4
    assert comp.shape == (4, 4)
    # Top-left 2×2 block is tile1 (value 8)
    assert np.all(comp[0:2, 0:2] == 8)
    # Bottom-right 2×2 block is tile0 (value 5)
    assert np.all(comp[2:4, 2:4] == 5)
