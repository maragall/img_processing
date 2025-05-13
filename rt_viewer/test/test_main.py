import argparse
from pathlib import Path

from rtviewer.datasource import DataSource
from rtviewer.cache import TileCache
from rtviewer.pyramid import PyramidBuilder
from rtviewer.stitcher_adapter import MISTAdapter
from rtviewer.renderer import TileRenderer
from rtviewer.controller import ViewerController


def main():
    parser = argparse.ArgumentParser(description="rtviewer CLI")
    parser.add_argument(
        "--dir", "-d", required=True, type=Path,
        help="Root directory of tile data"
    )
    parser.add_argument(
        "--mem", "-m", type=float, default=1e9,
        help="Maximum cache size in bytes"
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=4,
        help="Number of worker threads (stub, not yet used)"
    )
    args = parser.parse_args()

    root = args.dir
    ds = DataSource(root)
    cache = TileCache(datasource=ds, max_bytes=int(args.mem))
    pyr = PyramidBuilder(ds, [4, 8, 16])
    adapter = MISTAdapter()
    rend = TileRenderer()
    ctrl = ViewerController(ds, cache, pyr, adapter, rend)
    ctrl.run(root)


if __name__ == '__main__':
    main()
