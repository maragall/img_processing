#!/usr/bin/env python3
# rtviewer/controller.py

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import napari
from qtpy.QtCore import QTimer
from typing import Any, Dict, Tuple, List

import pandas as pd

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

        # Napari viewer and layer will be set in run()
        self.viewer: Any = None
        self.image_layer: Any = None

        # Precompute tile center positions (in pixel coords)
        self.tile_centers: Dict[int, Tuple[float, float]] = self.datasource.get_tile_centers()

        # Infer full grid dimensions from coordinates.csv (z=0)
        coords_csv = self.datasource.root / "0" / "coordinates.csv"
        df = pd.read_csv(coords_csv)
        xs = df["x (mm)"].unique()
        ys = df["y (mm)"].unique()
        self.grid_ncols = len(xs)
        self.grid_nrows = len(ys)

    def run(self, root: Path) -> None:
        """
        Launch Napari, show overview, and hook zoom & pan events.
        """
        self.datasource.root = root

        # Build overview pyramid levels
        levels = self.pyramid.build_levels()
        overview = levels[16]  # start at lowest resolution

        # Create Napari viewer and add the overview
        self.viewer = napari.Viewer()
        self.image_layer = self.viewer.add_image(
            overview,
            name="overview",
            multiscale=False,
            scale=[16, 16],
        )

        # Hook only the PUBLIC Napari camera events for zoom and pan.
        # Removed: canvas.events.camera_changed (private API, deprecated),
        #          and layer.events.scale / viewer.events.interactive,
        #          because they didn’t reliably fire in Napari 0.6+.
        self.viewer.camera.events.zoom.connect(self.on_view_changed)
        self.viewer.camera.events.center.connect(self.on_view_changed)

        napari.run()

    def on_view_changed(self, event: Any = None) -> None:
        """
        Called on zoom or pan: determine visible tiles, stitch them in background,
        and update the composite on the Qt main thread.
        """
        cam = self.viewer.camera
        center = cam.center  # (y, x)
        zoom = cam.zoom

        # Debug print to confirm the event fires
        print(f"[DEBUG] on_view_changed: center={center}, zoom={zoom:.2f}")

        # Compute the pixel‐space bounds of the viewport
        h, w = self.image_layer.data.shape
        half_h = (h / zoom) / 2.0
        half_w = (w / zoom) / 2.0
        ymin, ymax = center[0] - half_h, center[0] + half_h
        xmin, xmax = center[1] - half_w, center[1] + half_w

        # Figure out which FOVs lie inside the current view
        visible_fovs: List[int] = [
            fov for fov, (y_c, x_c) in self.tile_centers.items()
            if xmin <= x_c <= xmax and ymin <= y_c <= ymax
        ]

        # Load those tiles at full resolution
        tiles = [ self.cache.get(fov, 0, 1) for fov in visible_fovs ]

        # Submit the headless‐MIST stitch job
        future = self.executor.submit(
            self.adapter.align_tiles,
            tiles,
            self.grid_nrows,
            self.grid_ncols
        )

        def on_done(fut):
            # Remove any prior status overlay logic: we erased add_text entirely.
            # If you want UI feedback, use a Shapes/Text layer, but Napari Viewer.add_text
            # does not exist, so we stick to console logs.

            # Check for error
            if fut.exception():
                print(f"[ERROR] stitching failed: {fut.exception()}")
                return

            # Map tile_index → fov
            df = fut.result()
            df["fov"] = df["tile_index"].map(lambda idx: visible_fovs[idx])
            offsets = df[["fov", "dx", "dy"]]

            # Pick downsample factor
            if zoom < 6:
                level = 4
            elif zoom < 12:
                level = 8
            else:
                level = 16

            # Composite and update
            comp = self.renderer.composite(offsets, level)
            QTimer.singleShot(0, lambda: setattr(self.image_layer, "data", comp))

        future.add_done_callback(on_done)
