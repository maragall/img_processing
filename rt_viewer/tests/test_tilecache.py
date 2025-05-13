import numpy as np
import pytest
import threading
import time
from rtviewer.cache import TileCache
from rtviewer.datasource import DataSource  # Base class for type, but not initializing it

class DummyDataSource(DataSource):
    """A DataSource stub that does not require a real directory."""
    def __init__(self):
        # Skip DataSource.__init__; just track calls
        self.calls = []

    def load_tile(self, fov: int, z: int, level: int) -> np.ndarray:
        self.calls.append((fov, z, level))
        # return a 10×10 uint8 array (100 bytes)
        return np.zeros((10, 10), dtype=np.uint8)


def test_cache_hit_and_miss():
    ds = DummyDataSource()
    cache = TileCache(datasource=ds, max_bytes=200)  # can hold two 100-byte tiles

    # Miss on first access → DataSource.load_tile called once
    tile1 = cache.get(0, 0, 0)
    assert ds.calls == [(0, 0, 0)]
    assert tile1.shape == (10, 10)

    # Hit on second access → no additional load_tile call
    ds.calls.clear()
    tile1_again = cache.get(0, 0, 0)
    assert ds.calls == []
    assert np.array_equal(tile1, tile1_again)


def test_put_type_and_size_validation():
    ds = DummyDataSource()
    cache = TileCache(datasource=ds, max_bytes=500)

    # Non-array should raise
    with pytest.raises(TypeError):
        cache.put(0, 0, 0, "not-an-array")

    # Too-large array should raise
    big = np.zeros((1000, 1000), dtype=np.uint8)  # ~1 MB
    with pytest.raises(ValueError):
        cache.put(1, 0, 0, big)


def test_eviction_order():
    ds = DummyDataSource()
    cache = TileCache(datasource=ds, max_bytes=100)  # only one 100-byte tile fits

    # Load tile A → cached
    cache.get(0, 0, 0)
    assert (0, 0, 0) in cache.cache

    # Load tile B → eviction of A
    cache.get(1, 0, 0)
    assert (1, 0, 0) in cache.cache
    assert (0, 0, 0) not in cache.cache


def test_concurrent_access():
    """Simulate two threads racing to get the same tile."""
    ds = DummyDataSource()
    cache = TileCache(datasource=ds, max_bytes=200)

    # Monkey-patch load_tile to delay so threads overlap
    original_load = ds.load_tile
    def slow_load(fov, z, level):
        time.sleep(0.1)
        return original_load(fov, z, level)
    ds.load_tile = slow_load

    results = []

    def worker():
        results.append(cache.get(2, 0, 0))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()

    # Both threads should get an array and load_tile was only called once
    assert len(results) == 2
    assert ds.calls.count((2, 0, 0)) == 1
