import threading
from pathlib import Path
import pytest
import pandas as pd

import napari
from rtviewer.controller import ViewerController
from rtviewer.datasource import DataSource
from rtviewer.cache import TileCache
from rtviewer.pyramid import PyramidBuilder
from rtviewer.stitcher_adapter import MISTAdapter
from rtviewer.renderer import TileRenderer

# --- Helpers / Spies ---

class DummyEvents:
    def __init__(self, raise_on_connect=False):
        self.interactive = self
        self.scale = self
        self.connected_callbacks = []
        self.raise_on_connect = raise_on_connect

    def connect(self, callback):
        if self.raise_on_connect:
            raise RuntimeError("connect failed")
        self.connected_callbacks.append(callback)

class DummyLayer:
    def __init__(self, events):
        self.events = events
        self.data = None

class DummyViewer:
    def __init__(self, camera_events, layer_events):
        self.camera = type("C", (), {"events": camera_events})
        self._layer_events = layer_events
        self.bind_keys = {}
        self.added = []

    def add_image(self, overview, name, multiscale, scale):
        self.added.append({
            "overview": overview,
            "name": name,
            "multiscale": multiscale,
            "scale": scale,
        })
        return DummyLayer(self._layer_events)

    def bind_key(self, key):
        def decorator(fn):
            self.bind_keys[key] = fn
            return fn
        return decorator

# --- Fixtures ---

@pytest.fixture(autouse=True)
def stub_napari(monkeypatch):
    def viewer_factory():
        camera_events = DummyEvents(raise_on_connect=False)
        layer_events = DummyEvents()
        return DummyViewer(camera_events, layer_events)
    monkeypatch.setattr(napari, "Viewer", viewer_factory)
    monkeypatch.setattr(napari, "run", lambda: None)

@pytest.fixture(autouse=True)
def immediate_threads(monkeypatch):
    class ImmediateThread:
        def __init__(self, target, daemon):
            self._target = target
        def start(self):
            self._target()
    monkeypatch.setattr(threading, "Thread", ImmediateThread)

@pytest.fixture(autouse=True)
def stub_datasource(monkeypatch):
    monkeypatch.setattr(DataSource, '__init__', lambda self, root: setattr(self, 'root', root))

# --- Spies ---

class SpyAdapter(MISTAdapter):
    def __init__(self):
        super().__init__()
        self.called_with = None
    def align_tiles(self, tiles):
        self.called_with = tiles
        return super().align_tiles(tiles)

class SpyRenderer(TileRenderer):
    def __init__(self):
        super().__init__()
        self.calls = []
    def composite(self, offsets, level):
        self.calls.append((offsets, level))
        return super().composite(offsets, level)

# --- Tests ---

def test_run_attaches_camera_callback_and_updates(monkeypatch, tmp_path):
    ds = DataSource(root=tmp_path)
    cache = TileCache(ds, max_bytes=10**8)
    pyramid = PyramidBuilder(ds, [4, 8, 16])
    adapter = SpyAdapter()
    renderer = SpyRenderer()
    vc = ViewerController(ds, cache, pyramid, adapter, renderer)

    # Stub build_levels
    dummy_overview = object()
    monkeypatch.setattr(pyramid, "build_levels", lambda: {16: dummy_overview})

    vc.run(tmp_path)

    # Verify setup
    assert ds.root == tmp_path
    added = vc.viewer.added
    assert len(added) == 1
    call = added[0]
    assert call["overview"] is dummy_overview
    assert call["name"] == "overview"
    assert call["multiscale"] is False
    assert call["scale"] == [16, 16]

    # Invoke camera callback
    camera_events = vc.viewer.camera.events
    assert len(camera_events.connected_callbacks) == 1
    cb = camera_events.connected_callbacks[0]
    cb(None)

    # Check background work
    assert adapter.called_with == []
    # Inspect renderer calls
    assert len(renderer.calls) == 1
    offsets_df, lvl = renderer.calls[0]
    assert lvl == 16
    assert isinstance(offsets_df, pd.DataFrame)
    assert list(offsets_df.columns) == ['fov', 'dx', 'dy']
    assert len(offsets_df) == 4
    assert (offsets_df['dx'] == 0).all()
    assert (offsets_df['dy'] == 0).all()
    assert isinstance(vc.image_layer.data, type(renderer.composite(offsets_df, lvl)))


def test_run_fallback_to_layer_events(monkeypatch, tmp_path):
    ds = DataSource(root=tmp_path)
    cache = TileCache(ds, max_bytes=10**8)
    pyramid = PyramidBuilder(ds, [4, 8, 16])
    adapter = SpyAdapter()
    renderer = SpyRenderer()

    # Force camera.connect to fail
    def viewer_factory_fail():
        camera_events = DummyEvents(raise_on_connect=True)
        layer_events = DummyEvents()
        return DummyViewer(camera_events, layer_events)
    monkeypatch.setattr(napari, "Viewer", viewer_factory_fail)

    vc = ViewerController(ds, cache, pyramid, adapter, renderer)
    dummy_overview = object()
    monkeypatch.setattr(pyramid, "build_levels", lambda: {16: dummy_overview})

    vc.run(tmp_path)

    # Fallback hook
    layer_events = vc.image_layer.events
    assert len(layer_events.connected_callbacks) == 1
    fallback_cb = layer_events.connected_callbacks[0]
    fallback_cb()

    assert adapter.called_with == []
    assert len(renderer.calls) == 1
    offsets_df, lvl = renderer.calls[0]
    assert lvl == 16
    assert isinstance(offsets_df, pd.DataFrame)
    assert list(offsets_df.columns) == ['fov', 'dx', 'dy']
    assert len(offsets_df) == 4
    assert (offsets_df['dx'] == 0).all()
    assert (offsets_df['dy'] == 0).all()
    assert isinstance(vc.image_layer.data, type(renderer.composite(offsets_df, lvl)))
