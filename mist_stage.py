#!/usr/bin/env python3
"""
mist_stage.py — single-file, non-negotiable launch of NIST-MIST from Python.
Every public parameter, enum, and flag is driven off the bean's metadata so
you never mistype a flag.

Prerequisites
-------------
1. Fiji.app with the *MIST* update-site enabled.
2. `export FIJI_DIR=/absolute/path/to/Fiji.app`
3. `pip install imagej scyjava`

Run
---
    python mist_stage.py /absolute/path/to/0          # directory that holds the TIFF grid
"""

from __future__ import annotations 
import os, sys
from pathlib import Path
from typing import Any
import math
import logging
import threading
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# JVM bootstrap
# --------------------------------------------------------------------------- #
def _init_fiji() -> tuple[Any, Any]:
    try:
        import imagej, scyjava
        from jpype._jvmfinder import JVMNotFoundException
    except ImportError:
        sys.exit("pip install imagej scyjava")

    fiji_dir = Path(os.environ.get("FIJI_DIR", "")).expanduser()
    if not fiji_dir.is_dir():
        sys.exit("Set FIJI_DIR to a Fiji.app folder with MIST installed.")

    try:
        logger.info(f"Initializing ImageJ from {fiji_dir}")
        ij = imagej.init(fiji_dir.as_posix(), mode="headless")
        logger.info("ImageJ initialized successfully")
    except JVMNotFoundException as e:
        logger.error(f"Failed to initialize ImageJ: {str(e)}")
        sys.exit("No JDK found — install OpenJDK.\n" + str(e))

    return ij, scyjava

ij, sj = _init_fiji()

# --------------------------------------------------------------------------- #
#  Java classes
# --------------------------------------------------------------------------- #
StitchParams = sj.jimport("gov.nist.isg.mist.gui.params.StitchingAppParams")
MISTMain     = sj.jimport("gov.nist.isg.mist.MISTMain")
JStringArr   = sj.jimport("[Ljava.lang.String;")

# ENUMS
LoaderType       = sj.jimport("gov.nist.isg.mist.lib.tilegrid.loader.TileGridLoader$LoaderType")
GridOrigin       = sj.jimport("gov.nist.isg.mist.lib.tilegrid.loader.TileGridLoader$GridOrigin")
GridDirection    = sj.jimport("gov.nist.isg.mist.lib.tilegrid.loader.TileGridLoader$GridDirection")
BlendingMode     = sj.jimport("gov.nist.isg.mist.lib.export.BlendingMode")
CompressionMode  = sj.jimport("gov.nist.isg.mist.lib.export.CompressionMode")
Unit             = sj.jimport("gov.nist.isg.mist.lib.export.MicroscopyUnits")
StitchingType    = sj.jimport("gov.nist.isg.mist.lib.executor.StitchingExecutor$StitchingType")
LogType          = sj.jimport("gov.nist.isg.mist.lib.log.Log$LogType")
DebugType        = sj.jimport("gov.nist.isg.mist.lib.log.Debug$DebugType")
TransRefineType  = sj.jimport("gov.nist.isg.mist.lib.imagetile.Stitching$TranslationRefinementType")
FftwPlanType     = sj.jimport("gov.nist.isg.mist.lib.imagetile.fftw.FftwPlanType")

# --------------------------------------------------------------------------- #
#  Helper: bean → exact CLI args
# --------------------------------------------------------------------------- #
def bean_to_cli_args(jp: Any) -> list[str]:
    parts = {
        "InputParams":    jp.getInputParams(),
        "OutputParams":   jp.getOutputParams(),
        "AdvancedParams": jp.getAdvancedParams(),
        "LogParams":      jp.getLogParams(),
    }
    cli: list[str] = []
    for bean in parts.values():
        names = list(bean.getParameterNamesList())
        for name in names:
            # ← CHANGED: handle params whose getter suffix doesn't match the name
            cap = {
                "gridOrigin":       "Origin",
                "numberingPattern": "Numbering",
            }.get(name, name[0].upper() + name[1:])

            # pick the real getter/is-er
            if hasattr(bean, f"get{cap}"):
                val = getattr(bean, f"get{cap}")()
            elif hasattr(bean, f"is{cap}"):
                val = getattr(bean, f"is{cap}")()
            else:
                continue

            if val is None:
                continue
            if hasattr(val, "name"):
                val = val.name()

            # NaN → 'NaN'
            import math
            sval = "NaN" if isinstance(val, float) and math.isnan(val) else str(val)

            cli += [f"--{name}", sval]

    return cli


# --------------------------------------------------------------------------- #
#  Build fully-specified params bean (no JSON) — returns the bean itself
# --------------------------------------------------------------------------- #
def build_params(tile_dir: Path) -> Any:  # ← CHANGED: now returns StitchParams bean
    jp  = StitchParams()
    ip  = jp.getInputParams()
    op  = jp.getOutputParams()
    adv = jp.getAdvancedParams()
    lg  = jp.getLogParams()

    # ── INPUT ──────────────────────────────────────────────────────────────
    ip.setImageDir(tile_dir.as_posix())
    ip.setFilenamePattern("manual_r{rr}_c{cc}_0_Fluorescence_405_nm_Ex.tiff")
    ip.setFilenamePatternLoaderType(LoaderType.ROWCOL)
    ip.setGridWidth(8)
    ip.setGridHeight(11)
    ip.setOrigin(GridOrigin.UL)
    ip.setNumbering(GridDirection.HORIZONTALCOMBING)
    ip.setStartTileRow(0)
    ip.setStartTileCol(0)
    ip.setStartRow(0)
    ip.setStartCol(0)
    ip.setExtentWidth(8)
    ip.setExtentHeight(11)
    ip.setAssembleFromMetadata(False)
    ip.setAssembleNoOverlap(False)
    ip.setTimeSlicesEnabled(False)

    # ── OUTPUT ─────────────────────────────────────────────────────────────
    op.setOutputPath(str(tile_dir.parent))
    op.setOutFilePrefix("Fluo405_")
    op.setBlendingMode(BlendingMode.OVERLAY)
    op.setBlendingAlpha(float(0.0))
    op.setCompressionMode(CompressionMode.UNCOMPRESSED)
    op.setDisplayStitching(False)
    op.setOutputFullImage(False)
    op.setOutputMeta(True)
    op.setOutputImgPyramid(False)
    op.setPerPixelUnit(Unit.MICROMETER)
    op.setPerPixelX(7.52)
    op.setPerPixelY(7.52) 

    # ── ADVANCED ───────────────────────────────────────────────────────────
    adv.setProgramType(StitchingType.FFTW)
    adv.setUseDoublePrecision(True)
    adv.setNumCPUThreads(os.cpu_count())
    adv.setHorizontalOverlap(float("nan"))
    adv.setVerticalOverlap(float("nan"))
    adv.setOverlapUncertainty(float("nan"))
    adv.setNumFFTPeaks(0)
    adv.setRepeatability(0)
    adv.setTranslationRefinementType(TransRefineType.SINGLE_HILL_CLIMB)
    adv.setNumTranslationRefinementStartPoints(16)
    adv.setSuppressModalWarningDialog(True)
    adv.setUseBioFormats(False)
    adv.setEnableCudaExceptions(False)
    adv.setFftwPlanType(FftwPlanType.PATIENT)
    adv.setSaveFFTWPlan(True)
    adv.setLoadFFTWPlan(True)
    adv.setFftwLibraryName("libfftw3")
    adv.setFftwLibraryFileName("libfftw3.dylib")
    adv.setFftwLibraryPath(str(tile_dir.parent / "lib" / "fftw"))
    adv.setPlanPath(str(tile_dir.parent / "lib" / "fftw" / "fftPlans"))

    # ── LOGGING ────────────────────────────────────────────────────────────
    lg.setLogLevel(LogType.MANDATORY)
    lg.setDebugLevel(DebugType.NONE)

    return jp  # ← CHANGED: no more JSON, return bean

def run_mist_with_timeout(java_argv: Any, timeout_seconds: int = 3600) -> bool:
    """
    Run MIST with a timeout. Returns True if successful, False if timed out.
    """
    result = {"success": False, "error": None}
    
    def run_mist():
        try:
            logger.info("Starting MIST process")
            MISTMain.main(java_argv)
            result["success"] = True
            logger.info("MIST process completed successfully")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"MIST process failed: {str(e)}")
    
    thread = threading.Thread(target=run_mist)
    thread.daemon = True
    thread.start()
    
    # Wait for the thread to complete or timeout
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        logger.error(f"MIST process timed out after {timeout_seconds} seconds")
        return False
    
    if not result["success"]:
        logger.error(f"MIST process failed: {result['error']}")
        return False
    
    return True

def _run_one_channel(
    tile_dir: Path,
    channel: str,
    metadata: Path | None,
    assemble_from_metadata: bool
) -> bool:
    """
    Build the bean for a single channel and launch MISTMain with timeout.
    Returns True if successful, False if failed or timed out.
    """
    try:
        logger.info(f"Building parameters for channel {channel}")
        jp = build_params(tile_dir)

        # override filename pattern and prefix for this channel
        ip = jp.getInputParams()
        op = jp.getOutputParams()

        pattern = f"manual_r{{rr}}_c{{cc}}_0_Fluorescence_{channel}_nm_Ex.tiff"
        ip.setFilenamePattern(pattern)
        logger.info(f"Set filename pattern: {pattern}")

        prefix = f"Fluo{channel}_"
        op.setOutFilePrefix(prefix)
        logger.info(f"Set output prefix: {prefix}")

        if assemble_from_metadata:
            logger.info("Setting up metadata-based assembly")
            ip.setAssembleFromMetadata(True)
            if not (metadata and metadata.is_file()):
                logger.error(f"Metadata file not found: {metadata}")
                return False
            ip.setGlobalPositionsFile(metadata.as_posix())
            logger.info(f"Using metadata file: {metadata}")

        cli_args = bean_to_cli_args(jp)
        logger.info(f"Generated CLI arguments for channel {channel}")

        java_argv = JStringArr(len(cli_args))
        for i, a in enumerate(cli_args):
            java_argv[i] = a

        logger.info(f"Launching MIST for channel {channel}")
        success = run_mist_with_timeout(java_argv)
        
        if success:
            logger.info(f"Successfully completed channel {channel}")
        else:
            logger.error(f"Failed to complete channel {channel}")
        
        return success

    except Exception as e:
        logger.error(f"Error processing channel {channel}: {str(e)}")
        return False

# --------------------------------------------------------------------------- #
#  Entry
# --------------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python mist_stage.py /absolute/path/to/0")

    tile_dir = Path(sys.argv[1]).expanduser().resolve()
    if not tile_dir.is_dir():
        sys.exit("tile_dir must be a directory containing your TIFF grid.")

    jp = build_params(tile_dir)                         # ← CHANGED
    cli_args = bean_to_cli_args(jp)                     # ← CHANGED
    java_argv = JStringArr(len(cli_args))
    for i, arg in enumerate(cli_args):
        java_argv[i] = arg                              # ← CHANGED

    print("Launching MIST with exact bean-derived flags…")
    MISTMain.main(java_argv)                            # ← CHANGED
    print(" MIST stitching job started — tail Fiji's *Log* window.")

if __name__ == "__main__":
    main()