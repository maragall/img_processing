import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from rtviewer.controller import ViewerController
from rtviewer.datasource import DataSource
from rtviewer.cache import TileCache
from rtviewer.stitcher_adapter import MISTAdapter
from rtviewer.renderer import TileRenderer

# Dummy executor that runs tasks synchronously
class DummyFuture:
    def __init__(self, result):
        self._result = result
    def result(self):
        return self._result
    def add_done_callback(self, cb):
        cb(self)

class DummyExecutor:
    def submit(self, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return DummyFuture(result)

# Fake Napari viewer and layer to intercept image updates
class FakeLayer:
    def __init__(self, data):
        self.data = data
        # Dummy event stub
        class Evt: pass
        self.events = Evt()
        self.events.scale = Evt()
        self.events.scale.connect = lambda f: None

class FakeCamera:
    def __init__(self):
        # Center (y, x) in data coords and zoom factor
        self.center = (100.0, 100.0)
        self.zoom = 10.0
        class Evt:
            def connect(self, f): pass
        self.events = Evt()
        self.events.interactive = self.events

class FakeViewer:
    def __init__(self, initial_data):
        self.camera = FakeCamera()
        self._initial_data = initial_data
    def add_image(self, data, name, multiscale, scale):
        # Return a fake layer with data
        return FakeLayer(data)

# Fake DataSource with tile centers mapping
class FakeDataSource(DataSource):
    def __init__(self): pass
    def get_tile_centers(self):
        # One tile centered at (100,100)
        return {1: (100.0, 100.0)}

# Fake TileCache returning a constant tile image
class FakeCache(TileCache):
    def __init__(self): pass
    def get(self, fov, z, level):
        # Return a small constant array for any tile
        return np.ones((20,20), dtype=np.uint8)

# Fake MISTAdapter returning a known offset DataFrame
class FakeAdapter(MISTAdapter):
    def __init__(self): pass
    def align_tiles(self, tiles):
        # tiles is a list of numpy arrays; return DataFrame with fov and dx,dy
        return pd.DataFrame([{'fov': 1, 'dx': 5.0, 'dy': -3.0}])

# Fake TileRenderer producing a sentinel composite
class FakeRenderer(TileRenderer):
    def __init__(self): pass
    def composite(self, offsets, level):
        # Return array filled with level for testing
        shape = (int(200/level), int(200/level))
        return np.full(shape, fill_value=level, dtype=np.uint8)

@ pytest.fixture(autouse=True)
def patch_executor_and_view(monkeypatch):
    # Monkeypatch ThreadPoolExecutor to DummyExecutor
    monkeypatch.setattr('rtviewer.controller.ThreadPoolExecutor', lambda max_workers: DummyExecutor())
    # Monkeypatch QTimer.singleShot to call immediately
    monkeypatch.setattr('rtviewer.controller.QTimer.singleShot', lambda ms, fn: fn())

def test_on_view_changed_updates_layer(monkeypatch):
    # Create controller with fakes
    ds = FakeDataSource()
    cache = FakeCache()
    pyramid = None  # Not used in on_view_changed
    adapter = FakeAdapter()
    renderer = FakeRenderer()
    ctrl = ViewerController(ds, cache, pyramid, adapter, renderer)

    # Inject our FakeViewer and FakeLayer
    initial = np.zeros((200,200), dtype=np.uint8)
    fake_viewer = FakeViewer(initial)
    ctrl.viewer = fake_viewer
    ctrl.image_layer = fake_viewer.add_image(initial, name='test', multiscale=False, scale=[16,16])

    # Call the view-changed handler
    ctrl.on_view_changed()

    # After handler, layer.data should be updated to composite filled with correct level
    # zoom=10 -> level=8 (see on_view_changed logic: zoom<6->4, zoom<12->8, else->16)
    expected = np.full((int(200/8), int(200/8)), fill_value=8, dtype=np.uint8)
    assert np.array_equal(ctrl.image_layer.data, expected)
