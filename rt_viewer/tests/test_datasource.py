# tests/test_datasource_real.py
import json
import math
from pathlib import Path

import numpy as np
import dask.array as da
import pytest
import tifffile

from rtviewer.datasource import DataSource, FNAME_RE


def make_tile(path: Path, value: int, shape=(2, 3)):
    """Write a single‐channel TIFF filled with `value`."""
    arr = np.full(shape, fill_value=value, dtype=np.uint16)
    tifffile.imwrite(path, arr)


@pytest.fixture
def tmp_dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    # 1) valid acquisition parameters JSON
    params = {"sensor_pixel_size_um": 0.752}
    (root / "acquisition parameters.json").write_text(json.dumps(params))
    # 2) create 4 FOVs, each with two channels 'A','B' at z=0
    fovs = [1, 2, 3, 4]
    shape = (2, 3)
    for fov in fovs:
        for suffix, mult in [("A", 1), ("B", 10)]:
            fname = f"manual_{fov}_0_{suffix}.tiff"
            make_tile(root / fname, value=fov * mult, shape=shape)
    return root, fovs, shape


def test_fname_regex_matches_expected():
    # sanity‐check your regex
    m = FNAME_RE.match("manual_12_3_Fluorescence_405_nm_Ex.tiff")
    assert m
    assert m.group("fov") == "12"
    assert m.group("z") == "3"
    assert m.group("suffix") == "Fluorescence_405_nm_Ex"


def test_init_errors_on_missing_root(tmp_path):
    fake = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError):
        DataSource(fake)


def test_init_errors_on_missing_json(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        DataSource(root)


def test_init_errors_on_malformed_json(tmp_path):
    root = tmp_path / "badjson"
    root.mkdir()
    (root / "acquisition parameters.json").write_text("{ not valid }")
    with pytest.raises(ValueError):
        DataSource(root)


def test_load_tile_stack_and_dtype(tmp_dataset):
    root, fovs, shape = tmp_dataset
    ds = DataSource(root)
    # pick fov=2, z=0
    arr = ds.load_tile(fov=2, z=0, level=1)
    # must be shape (C, H, W) with C==2, H,W from shape
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (2, *shape)
    # channel order sorted by suffix: 'A'→value=2, 'B'→value=20
    np.testing.assert_array_equal(arr[0], np.full(shape, 2, dtype=np.uint16))
    np.testing.assert_array_equal(arr[1], np.full(shape, 20, dtype=np.uint16))


def test_load_tile_errors_on_missing_tile(tmp_dataset):
    root, _, _ = tmp_dataset
    ds = DataSource(root)
    with pytest.raises(FileNotFoundError):
        ds.load_tile(fov=99, z=0, level=1)


def test_load_tile_errors_on_unsupported_level(tmp_dataset):
    root, _, _ = tmp_dataset
    ds = DataSource(root)
    with pytest.raises(ValueError):
        ds.load_tile(fov=1, z=0, level=2)


def test_load_tile_errors_on_corrupted_tiff(tmp_dataset, tmp_path, monkeypatch):
    root, fovs, shape = tmp_dataset
    ds = DataSource(root)
    # point one tile path to a zero‐byte file
    bad = root / f"manual_{fovs[0]}_0_A.tiff"
    bad.write_bytes(b"")
    with pytest.raises(IOError):
        ds.load_tile(fov=fovs[0], z=0, level=1)


def test_load_overview_shape_and_content(tmp_dataset):
    root, fovs, shape = tmp_dataset
    ds = DataSource(root)
    mosaic = ds.load_overview(level=1)
    assert isinstance(mosaic, da.Array)
    arr = mosaic.compute()
    # grid: 4 tiles → cols=ceil(sqrt(4))=2, rows=2
    tile_h, tile_w = shape
    expected_shape = (2 * tile_h, 2 * tile_w)
    assert arr.shape == expected_shape
    # verify each block == first channel value (i.e. fov number)
    for idx, fov in enumerate(sorted(fovs)):
        r = idx // 2
        c = idx % 2
        block = arr[r * tile_h:(r + 1) * tile_h,
                    c * tile_w:(c + 1) * tile_w]
        expected = np.full(shape, fov, dtype=np.uint16)
        np.testing.assert_array_equal(block, expected)


def test_load_overview_errors_on_unsupported_level(tmp_dataset):
    root, _, _ = tmp_dataset
    ds = DataSource(root)
    with pytest.raises(ValueError):
        ds.load_overview(level=2)
