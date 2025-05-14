from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import napari
from qtpy.QtCore import QTimer
from typing import Any, Dict, Tuple

from .datasource import DataSource
from .cache import TileCache
from .pyramid import PyramidBuilder
from .stitcher_adapter import MISTAdapter
from .renderer import TileRenderer


class ViewerController:
    """
    Controller tying DataSource, TileCache, PyramidBuilder, MISTAdapter, and TileRenderer
    into a Napari viewer that stitches tiles on zoom/pan events in real time.
    """
    def __init__(
        self,
        datasource: DataSource,
        cache: TileCache,
        pyramid: PyramidBuilder,
        adapter: MISTAdapter,
        renderer: TileRenderer,
        max_workers: int = 4,
    ) -> None:
        """
        Initialize controller with components and a thread pool for background stitching.
        """
        self.datasource = datasource
        self.cache = cache
        self.pyramid = pyramid
        self.adapter = adapter
        self.renderer = renderer
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.viewer: Any = None
        self.image_layer: Any = None
        # Precompute tile center positions (in pixel coords) for visibility tests
        # expects DataSource.get_tile_centers() -> Dict[fov, (y_center_px, x_center_px)]
        self.tile_centers: Dict[int, Tuple[float, float]] = self.datasource.get_tile_centers()

    def run(self, root: Path) -> None:
        """
        Launch Napari, show overview, and hook zoom/pan events.
        """
        self.datasource.root = root
        levels = self.pyramid.build_levels()
        self.overview_levels = levels
        # Use level=16 overview to display initially
        overview = levels[16]

        self.viewer = napari.Viewer()
        self.image_layer = self.viewer.add_image(
            overview,
            name="overview",
            multiscale=False,
            scale=[16, 16],
        )

        # Hook camera and layer events to on_view_changed
        try:
            self.viewer.camera.events.interactive.connect(self.on_view_changed)
        except Exception:
            self.image_layer.events.scale.connect(self.on_view_changed)

        napari.run()

    def on_view_changed(self, event: Any = None) -> None:
        """
        Called on zoom/pan: determine visible tiles, stitch them in background,
        and update composite on Qt main thread when done.
        """
        # Determine current view center and zoom
        cam = self.viewer.camera
        center = cam.center  # (y_center, x_center)
        zoom = cam.zoom      # scale factor

        # Determine half-window in data pixels
        h, w = self.image_layer.data.shape
        half_h = (h / zoom) / 2.0
        half_w = (w / zoom) / 2.0
        ymin = center[0] - half_h
        ymax = center[0] + half_h
        xmin = center[1] - half_w
        xmax = center[1] + half_w

        # Determine which FOVs are visible in this view
        visible_fovs = []
        for fov, (y_c, x_c) in self.tile_centers.items():
            if xmin <= x_c <= xmax and ymin <= y_c <= ymax:
                visible_fovs.append(fov)

        # Load tile arrays for each visible fov at full resolution
        tiles = []
        for fov in visible_fovs:
            tile_arr = self.cache.get(fov, 0, 1)  # z=0, level=1
            tiles.append(tile_arr)

        # Submit background stitching job
        future = self.executor.submit(self.adapter.align_tiles, tiles)

        def on_done(fut):
            offsets = fut.result()  # DataFrame with columns ['fov','dx','dy']
            # Choose composite level based on zoom thresholds
            if zoom < 6:
                level = 4
            elif zoom < 12:
                level = 8
            else:
                level = 16
            composite = self.renderer.composite(offsets, level)
            # Update layer data on Qt main thread
            QTimer.singleShot(0, lambda: setattr(self.image_layer, 'data', composite))

        future.add_done_callback(on_done)