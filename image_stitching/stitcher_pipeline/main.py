#!/usr/bin/env python3
"""
stitcher_pipeline.main  –  Command-line entry point for the TIFF-tile pipeline.

Run `python -m stitcher_pipeline.main --help` for usage.

Sub-commands:
  rename          Zero-pad FOV indices in TIFF names + coordinates.csv
  generate-params Rename to row/col, pad blanks, print MIST settings
  uniformize      Force every TIFF to the same pixel shape
  full            Run all three stages in sequence
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stitcher_pipeline.rename_stage import rename_stage
from stitcher_pipeline.generate_stage import generate_stage
from stitcher_pipeline.uniformize_stage import uniformize_stage


# ---------------------------------------------------------------- CLI builder
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified TIFF-tile preprocessing pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # helper to avoid repetition
    def _add(name: str, func, help_: str):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(handler=func)
        sp.add_argument(
            "--dir", required=True, type=Path,
            help="directory containing coordinates.csv and TIFF tiles",
        )

    _add("rename",          rename_stage,
         "zero-pad FOV indices in filenames + CSV")
    _add("generate-params", generate_stage,
         "rename to row/col, pad blanks, emit MIST settings")
    _add("uniformize",      uniformize_stage,
         "enforce uniform tile shape")

    # convenience wrapper
    full = sub.add_parser("full", help="run rename → generate-params → uniformize")
    full.add_argument("--dir", required=True, type=Path)

    return p


# ----------------------------------------------------------------- entrypoint
def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # "full" runs all stages in canonical order
    if args.cmd == "full":
        for stage in (rename_stage, generate_stage, uniformize_stage):
            stage(args.dir)
    else:
        # any single-stage command
        args.handler(args.dir)


if __name__ == "__main__":  # pragma: no cover
    main()