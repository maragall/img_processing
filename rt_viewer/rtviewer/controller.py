from pathlib import Path
import threading
import napari
from typing import Any

from .datasource import DataSource
from .cache import TileCache
from .pyramid import PyramidBuilder
from .stitcher_adapter import MISTAdapter
from .renderer import TileRenderer


class ViewerController:
    """
    Controller tying together data source, cache, pyramid, adapter, and renderer
    with a Napari viewer. Stubbed callbacks to run without real data.
    """
    def __init__(
        self,
        datasource: DataSource,
        cache: TileCache,
        pyramid: PyramidBuilder,
        adapter: MISTAdapter,
        renderer: TileRenderer,
    ) -> None:
        """
        Initialize with all components.
        """
        self.datasource = datasource
        self.cache = cache
        self.pyramid = pyramid
        self.adapter = adapter
        self.renderer = renderer
        self.viewer: Any = None
        self.image_layer: Any = None

    def run(self, root: Path) -> None:
        """
        Launch Napari viewer, display overview at level=16, and stub event hooks.

        Parameters
        ----------
        root : Path
            Root directory for the data source.
        """
        # Point data source to root
        self.datasource.root = root

        # Build pyramid levels and grab overview at level 16
        levels = self.pyramid.build_levels()
        overview = levels.get(16)

        # Initialize viewer and add multiscale overview
        self.viewer = napari.Viewer()
        self.image_layer = self.viewer.add_image(
            overview,
            name="overview",
            multiscale=False,
            scale=[16, 16],  # placeholder scales
        )

        # Keybindings for zoom levels 4, 8, 16
        @self.viewer.bind_key('4')
        def _zoom4(viewer):
            # TODO: set viewer.camera.zoom or layer.scale accordingly
            print("Zoom to level 4 (stub)")

        @self.viewer.bind_key('8')
        def _zoom8(viewer):
            print("Zoom to level 8 (stub)")

        @self.viewer.bind_key('1')  # using '1' for '16' stub
        def _zoom16(viewer):
            print("Zoom to level 16 (stub)")

        # Camera change callback stub: re-align tiles and update composite
        def _on_camera_change(event=None):
            def _worker():
                # stub: align no tiles, composite stub image
                offsets = self.adapter.align_tiles([])
                composite = self.renderer.composite(offsets, level=16)
                self.image_layer.data = composite
                print("Updated composite (stub)")
            threading.Thread(target=_worker, daemon=True).start()

        # Connect camera events (stubbed)
        try:
            self.viewer.camera.events.interactive.connect(_on_camera_change)
        except Exception:
            # Fallback: connect to layer events
            self.image_layer.events.scale.connect(_on_camera_change)

        napari.run()
