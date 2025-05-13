#!/usr/bin/env python3
"""
stitcher_pipeline.main – unified CLI dispatcher with metadata‐driven multi‐channel runs.

Sub-commands
------------
rename            : zero-pad FOV indices
generate-params   : row/col rename, blank padding, echo MIST settings
uniformize        : enforce uniform tile shape
run-mist          : stitch (with optional assemble-from-metadata)
full              : rename → generate-params → uniformize → multi‐channel stitching
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import logging

from stitcher_pipeline.rename_stage import rename_stage
from stitcher_pipeline.generate_stage import generate_stage
from stitcher_pipeline.uniformize_stage import uniformize_stage

# Import the API from mist_stage
from stitcher_pipeline.mist_stage import (
    build_params,
    bean_to_cli_args,
    MISTMain,
    JStringArr,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHANNELS = ["405"]#, "488", "561", "638"]


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
    _add(sub, "run-mist", "stitch via MIST (one channel)")

    full_p = _add(sub, "full",
                  "rename → generate-params → uniformize → multi‐channel stitching")
    return p


def _run_one_channel(
    tile_dir: Path,
    channel: str,
    metadata: Path | None,
    assemble_from_metadata: bool
):
    """
    Build the bean for a single channel, optionally turning on
    assembleFromMetadata + globalPositionsFile, then launch MISTMain.
    """
    logger.info(f"Starting processing for channel {channel}")
    jp = build_params(tile_dir)
    logger.info("Built parameters successfully")

    # override filename pattern and prefix for this channel
    ip = jp.getInputParams()
    op = jp.getOutputParams()

    pattern = f"manual_r{{rr}}_c{{cc}}_0_Fluorescence_{channel}_nm_Ex.tiff"
    ip.setFilenamePattern(pattern)
    logger.info(f"Set filename pattern: {pattern}")

    prefix = f"Fluo{channel}_"
    op.setOutFilePrefix(prefix)
    logger.info(f"Set output prefix: {prefix}")

    # assemble-from-metadata?
    if assemble_from_metadata:
        logger.info("Setting up metadata-based assembly")
        ip.setAssembleFromMetadata(True)
        if not (metadata and metadata.is_file()):
            logger.error(f"Metadata file not found: {metadata}")
            sys.exit(f"Metadata file not found: {metadata}")
        ip.setGlobalPositionsFile(metadata.as_posix())
        logger.info(f"Using metadata file: {metadata}")

    # debug dump
    print(f"\n=== CHANNEL {channel} {'(from metadata)' if assemble_from_metadata else ''} ===")
    cli_args = bean_to_cli_args(jp)
    print("CLI ARGS:", cli_args)

    # invoke MISTMain
    logger.info("Preparing to launch MISTMain")
    java_argv = JStringArr(len(cli_args))
    for i, a in enumerate(cli_args):
        java_argv[i] = a

    logger.info("Launching MISTMain...")
    try:
        MISTMain.main(java_argv)
        logger.info(f"Successfully completed channel {channel}")
        return True
    except Exception as e:
        logger.error(f"Error processing channel {channel}: {str(e)}")
        return False


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.cmd == "full":
        tile_dir = args.dir
        logger.info(f"Starting full pipeline in directory: {tile_dir}")

        # 1) Preprocess
        logger.info("Starting rename stage")
        rename_stage(tile_dir)
        logger.info("Starting generate stage")
        mist_dict = generate_stage(tile_dir)
        logger.info("Starting uniformize stage")
        uniformize_stage(tile_dir)

        # 2) Stitch 405 nm first (no metadata)
        logger.info("Starting first channel (405) processing")
        if not _run_one_channel(tile_dir, "405", None, assemble_from_metadata=False):
            logger.error("Failed to process first channel (405). Aborting pipeline.")
            sys.exit(1)

        # compute metadata path (parent dir / Fluo405_global-positions-1.txt)
        parent = tile_dir.parent
        meta_file = parent / "Fluo405_global-positions-1.txt"
        logger.info(f"Looking for metadata file at: {meta_file}")

        if not meta_file.exists():
            logger.error(f"Metadata file not found at {meta_file}. Aborting pipeline.")
            sys.exit(1)

        # 3) Stitch other channels using that metadata
        failed_channels = []
        for ch in CHANNELS[1:]:
            logger.info(f"Starting processing for channel {ch}")
            if not _run_one_channel(
                tile_dir,
                ch,
                metadata=meta_file,
                assemble_from_metadata=True
            ):
                logger.error(f"Failed to process channel {ch}")
                failed_channels.append(ch)
            else:
                logger.info(f"Completed processing for channel {ch}")

        if failed_channels:
            logger.error(f"The following channels failed to process: {', '.join(failed_channels)}")
            sys.exit(1)
        else:
            logger.info("All channels processed successfully")

    elif args.cmd == "rename":
        rename_stage(args.dir)

    elif args.cmd == "generate-params":
        generate_stage(args.dir)

    elif args.cmd == "uniformize":
        uniformize_stage(args.dir)

    elif args.cmd == "run-mist":
        sys.exit("Use 'full' for multi‐channel automated runs.")

    else:
        sys.exit("Unknown command")

if __name__ == "__main__":
    main()
