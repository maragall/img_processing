import argparse
from pathlib import Path

from .datasource import DataSource
from .cache import TileCache
from .pyramid import PyramidBuilder
from .stitcher_adapter import MISTAdapter
from .renderer import TileRenderer
from .controller import ViewerController


def main():
    parser = argparse.ArgumentParser(description="rtviewer CLI")
    parser.add_argument(
        "--dir", "-d", required=True, type=Path,
        help="Root directory of tile data"
    )
    parser.add_argument(
        "--mem", "-m", type=int, default=1_000_000_000,
        help="Maximum cache size in bytes"
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=4,
        help="Number of worker threads (stub, not yet used)"
    )
    args = parser.parse_args()

    root = args.dir
    max_bytes = args.mem
    # threads currently not used in stubs
    threads = args.threads

    # Initialize components
    ds = DataSource(root)
    cache = TileCache(ds, max_bytes)
    pyramid = PyramidBuilder(ds, [4, 8, 16])
    adapter = MISTAdapter()
    renderer = TileRenderer()

    # Launch viewer
    controller = ViewerController(ds, cache, pyramid, adapter, renderer)
    controller.run(root)


if __name__ == "__main__":
    main()
