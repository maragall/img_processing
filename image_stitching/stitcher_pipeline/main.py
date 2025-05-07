#!/usr/bin/env python3
"""
stitcher_pipeline.main – unified CLI dispatcher.

Sub-commands
------------
rename            : zero-pad FOV indices
generate-params   : row/col rename, blank padding, echo MIST settings
uniformize        : enforce uniform tile shape
run-mist          : *only* execute mist_stage.py (expects pre-processed dir)
full              : rename → generate-params → uniformize → run-mist
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

from stitcher_pipeline.rename_stage import rename_stage
from stitcher_pipeline.generate_stage import generate_stage
from stitcher_pipeline.uniformize_stage import uniformize_stage

# ---------------------------------------------------------------- CLI builder

def _add(sub, name: str, help_: str):
    p = sub.add_parser(name, help=help_)
    p.add_argument(
        "--dir", required=True, type=Path,
        help="directory containing coordinates.csv and TIFF tiles"
    )
    return p

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TIFF-tile preprocessing & MIST runner"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    _add(sub, "rename", "zero-pad FOV indices in filenames + CSV")
    _add(sub, "generate-params",
         "rename to row/col, pad blanks, print MIST parameters")
    _add(sub, "uniformize", "enforce uniform tile shape")
    _add(sub, "run-mist", "execute mist_stage.py via PyImageJ")

    # convenience wrapper
    _add(sub, "full", "rename → generate-params → uniformize → run-mist")
    return p

# ---------------------------------------------------------------- helpers

def _run_mist_stage(tile_dir: Path) -> None:
    """
    Invoke mist_stage.py as a separate process.
    Expects mist_stage.py next to this file.
    """
    script = Path(__file__).parent / "mist_stage.py"
    if not script.exists():
        sys.exit(f"❌  Cannot find {script!r}")
    cmd = [sys.executable, str(script), str(tile_dir)]
    subprocess.run(cmd, check=True)

# ---------------------------------------------------------------- entrypoint

def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.cmd == "rename":
        rename_stage(args.dir)

    elif args.cmd == "generate-params":
        generate_stage(args.dir)

    elif args.cmd == "uniformize":
        uniformize_stage(args.dir)

    elif args.cmd == "run-mist":
        _run_mist_stage(args.dir)

    elif args.cmd == "full":
        rename_stage(args.dir)
        generate_stage(args.dir)
        uniformize_stage(args.dir)
        _run_mist_stage(args.dir)

if __name__ == "__main__":
    main()
