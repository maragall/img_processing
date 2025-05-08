#!/usr/bin/env python3
"""
stitcher_pipeline.main – unified CLI dispatcher with metadata support for MIST.

Sub-commands
------------
rename            : zero-pad FOV indices
generate-params   : row/col rename, blank padding, echo MIST settings
uniformize        : enforce uniform tile shape
run-mist          : stitch (with optional assemble-from-metadata)
full              : rename → generate-params → uniformize → run-mist
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from stitcher_pipeline.rename_stage import rename_stage
from stitcher_pipeline.generate_stage import generate_stage
from stitcher_pipeline.uniformize_stage import uniformize_stage

# Import the new API from mist_stage
from stitcher_pipeline.mist_stage import (
    build_params,
    bean_to_cli_args,
    MISTMain,
    JStringArr,
)


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

    # for run-mist and full, also accept a metadata file
    run_p = _add(sub, "run-mist", "stitch via MIST, optionally from metadata")
    run_p.add_argument(
        "--metadata", required=False, type=Path,
        help="global-positions file to assemble from metadata"
    )

    full_p = _add(sub, "full",
                  "rename → generate-params → uniformize → run-mist")
    full_p.add_argument(
        "--metadata", required=False, type=Path,
        help="global-positions file to assemble from metadata"
    )

    return p


def _run_mist_direct(tile_dir: Path, metadata: Path | None = None):
    # build the StitchParams bean
    jp = build_params(tile_dir)

    # if metadata given, enable assemble-from-metadata
    if metadata:
        if not metadata.is_file():
            sys.exit(f"❌  Metadata file not found: {metadata}")
        ip = jp.getInputParams()
        ip.setAssembleFromMetadata(True)
        ip.setGlobalPositionsFile(metadata.as_posix())

    # dump for debugging
    print("\n--- BEAN PARAMETER NAMES ---")
    for section, getter in (
        ("InputParams",     jp.getInputParams),
        ("OutputParams",    jp.getOutputParams),
        ("AdvancedParams",  jp.getAdvancedParams),
        ("LogParams",       jp.getLogParams),
    ):
        bean = getter()
        names = list(bean.getParameterNamesList())
        print(f"{section}: {names}")

    cli_args = bean_to_cli_args(jp)
    print("\n--- FINAL CLI ARGS for MISTMain ---")
    print(cli_args)
    print("\nChecking args for stitching:")
    # you could add further validation here if desired
    print("Arg check passed")

    # invoke MISTMain
    java_argv = JStringArr(len(cli_args))
    for i, a in enumerate(cli_args):
        java_argv[i] = a

    print("\n🚀 Launching MISTMain with above flags…\n")
    MISTMain.main(java_argv)
    print("\n🎉  Done.\n")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.cmd == "rename":
        rename_stage(args.dir)

    elif args.cmd == "generate-params":
        generate_stage(args.dir)

    elif args.cmd == "uniformize":
        uniformize_stage(args.dir)

    elif args.cmd == "run-mist":
        _run_mist_direct(args.dir, getattr(args, "metadata", None))

    elif args.cmd == "full":
        rename_stage(args.dir)
        generate_stage(args.dir)
        uniformize_stage(args.dir)
        _run_mist_direct(args.dir, getattr(args, "metadata", None))


if __name__ == "__main__":
    main()
