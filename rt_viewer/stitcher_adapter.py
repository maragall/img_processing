#!/usr/bin/env python3
"""
mist_adapter.py — Adapter for headless, in-memory MIST stitching
using exactly the same parameters as mist_stage.py.
"""

from typing import List, Tuple
from pathlib import Path
import os
import numpy as np
import pandas as pd
import jpype
from jpype import JArray, JByte, JShort, JFloat, JClass


class MISTAdapter:
    """
    Adapter for headless, in-memory MIST stitching.
    """

    def __init__(self, tile_dir: str, fiji_dir: str = None) -> None:
        """
        Start JVM (if needed), import all MIST & ImageJ classes,
        and record the directory holding your TIFF grid.

        Parameters
        ----------
        tile_dir : str
            Path to the directory containing your TIFF grid (e.g., the '0' folder).
        fiji_dir : str, optional
            Path to the directory containing your MIST & ImageJ JARs (e.g., Fiji.app root).
            If None, will use FIJI_DIR environment variable.
        """
        self.tile_dir = Path(tile_dir).expanduser().resolve()

        if fiji_dir is None:
            fiji_dir = os.environ.get("FIJI_DIR")
        if not fiji_dir:
            raise ValueError(
                "FIJI_DIR environment variable not set and no fiji_dir argument provided."
            )
        fiji_dir = Path(fiji_dir).expanduser().resolve()

        if not jpype.isJVMStarted():
            # Recursively find all jars under the given directory
            jars = []
            for root, _, files in os.walk(fiji_dir):
                for fname in files:
                    if fname.lower().endswith(".jar"):
                        jars.append(str(Path(root) / fname))
            if not jars:
                raise RuntimeError(f"No .jar files found under '{fiji_dir}'")
            classpath = os.pathsep.join(jars)
            jpype.startJVM("-ea", f"-Djava.class.path={classpath}")

        self._import_java_classes()

    def _import_java_classes(self) -> None:
        # Core MIST classes
        self.StitchingAppParams = JClass("gov.nist.isg.mist.gui.params.StitchingAppParams")
        self.MISTMain               = JClass("gov.nist.isg.mist.MISTMain")
        self.TileGrid               = JClass("gov.nist.isg.mist.lib.tilegrid.TileGrid")
        self.ImageTileClass         = JClass("gov.nist.isg.mist.lib.imagetile.ImageTile")

        # ImageJ processors
        self.ByteProcessor  = JClass("ij.process.ByteProcessor")
        self.ShortProcessor = JClass("ij.process.ShortProcessor")
        self.FloatProcessor = JClass("ij.process.FloatProcessor")
        self.File           = JClass("java.io.File")

        # Enums & other beans to match mist_stage.py
        self.LoaderType       = JClass("gov.nist.isg.mist.lib.tilegrid.loader.TileGridLoader$LoaderType")
        self.GridOrigin       = JClass("gov.nist.isg.mist.lib.tilegrid.loader.TileGridLoader$GridOrigin")
        self.GridDirection    = JClass("gov.nist.isg.mist.lib.tilegrid.loader.TileGridLoader$GridDirection")
        self.BlendingMode     = JClass("gov.nist.isg.mist.lib.export.BlendingMode")
        self.CompressionMode  = JClass("gov.nist.isg.mist.lib.export.CompressionMode")
        self.Unit             = JClass("gov.nist.isg.mist.lib.export.MicroscopyUnits")
        self.StitchingType    = JClass("gov.nist.isg.mist.lib.executor.StitchingExecutor$StitchingType")
        self.LogType          = JClass("gov.nist.isg.mist.lib.log.Log$LogType")
        self.DebugType        = JClass("gov.nist.isg.mist.lib.log.Debug$DebugType")
        self.TransRefineType  = JClass("gov.nist.isg.mist.lib.imagetile.Stitching$TranslationRefinementType")
        self.FftwPlanType     = JClass("gov.nist.isg.mist.lib.imagetile.fftw.FftwPlanType")

        # Reflection into private ImageTile fields
        cls = self.ImageTileClass.class_
        self._pixels = cls.getDeclaredField("pixels")
        self._width  = cls.getDeclaredField("width")
        self._height = cls.getDeclaredField("height")
        self._loaded = cls.getDeclaredField("pixelsLoaded")
        for f in (self._pixels, self._width, self._height, self._loaded):
            f.setAccessible(True)

    def align_tiles(
        self,
        tiles: List[np.ndarray],
        num_rows: int,
        num_cols: int,
        overlap_pct: float = 20.0,
        downsample: int = 2,
    ) -> pd.DataFrame:
        """
        Align a regular grid of numpy tiles via MIST, using exactly the same
        bean parameters as mist_stage.py.

        Returns a DataFrame with columns ['tile_index', 'dx', 'dy'].
        """
        # 1. build fully-specified params bean
        jp = self.StitchingAppParams()
        ip = jp.getInputParams()
        op = jp.getOutputParams()
        adv = jp.getAdvancedParams()
        lg = jp.getLogParams()

        # ── INPUT ──────────────────────────────────────────────────────────────
        ip.setImageDir(self.tile_dir.as_posix())
        ip.setFilenamePattern("manual_r{rr}_c{cc}_0_Fluorescence_405_nm_Ex.tiff")
        ip.setFilenamePatternLoaderType(self.LoaderType.ROWCOL)
        ip.setGridWidth(num_cols)
        ip.setGridHeight(num_rows)
        ip.setOrigin(self.GridOrigin.UL)
        ip.setNumbering(self.GridDirection.HORIZONTALCOMBING)
        ip.setStartTileRow(0)
        ip.setStartTileCol(0)
        ip.setStartRow(0)
        ip.setStartCol(0)
        ip.setExtentWidth(num_cols)
        ip.setExtentHeight(num_rows)
        ip.setAssembleFromMetadata(False)
        ip.setAssembleNoOverlap(False)
        ip.setTimeSlicesEnabled(False)

        # ── OUTPUT ─────────────────────────────────────────────────────────────
        op.setOutputPath(str(self.tile_dir.parent))
        op.setOutFilePrefix("Fluo405_")
        op.setBlendingMode(self.BlendingMode.OVERLAY)
        op.setBlendingAlpha(float(0.0))
        op.setCompressionMode(self.CompressionMode.UNCOMPRESSED)
        op.setDisplayStitching(False)
        op.setOutputFullImage(False)
        op.setOutputMeta(True)
        op.setOutputImgPyramid(False)
        op.setPerPixelUnit(self.Unit.MICROMETER)
        op.setPerPixelX(7.52)
        op.setPerPixelY(7.52)

        # ── ADVANCED ───────────────────────────────────────────────────────────
        adv.setProgramType(self.StitchingType.FFTW)
        adv.setUseDoublePrecision(True)
        adv.setNumCPUThreads(os.cpu_count())
        adv.setHorizontalOverlap(float("nan"))
        adv.setVerticalOverlap(float("nan"))
        adv.setOverlapUncertainty(float("nan"))
        adv.setNumFFTPeaks(0)
        adv.setRepeatability(0)
        adv.setTranslationRefinementType(self.TransRefineType.SINGLE_HILL_CLIMB)
        adv.setNumTranslationRefinementStartPoints(16)
        adv.setSuppressModalWarningDialog(True)
        adv.setUseBioFormats(False)
        adv.setEnableCudaExceptions(False)
        adv.setFftwPlanType(self.FftwPlanType.PATIENT)
        adv.setSaveFFTWPlan(True)
        adv.setLoadFFTWPlan(True)
        adv.setFftwLibraryName("libfftw3")
        adv.setFftwLibraryFileName("libfftw3.dylib")
        adv.setFftwLibraryPath(str(self.tile_dir.parent / "lib" / "fftw"))
        adv.setPlanPath(str(self.tile_dir.parent / "lib" / "fftw" / "fftPlans"))

        # ── LOGGING ────────────────────────────────────────────────────────────
        lg.setLogLevel(self.LogType.MANDATORY)
        lg.setDebugLevel(self.DebugType.NONE)

        # 2. wrap numpy arrays as ImageJ processors
        tile_images = {}
        for idx, img in enumerate(tiles):
            h, w = img.shape[:2]
            dtype = img.dtype

            if dtype == np.uint8:
                arr = JArray(JByte)(img.ravel().tolist())
                iproc = self.ByteProcessor(w, h, arr, None)
            elif dtype == np.uint16:
                arr = JArray(JShort)(img.ravel().tolist())
                iproc = self.ShortProcessor(w, h, arr, None)
            elif np.issubdtype(dtype, np.floating):
                arr = JArray(JFloat)(img.astype("float32").ravel().tolist())
                iproc = self.FloatProcessor(w, h, arr, None)
            else:
                norm8 = (img / img.max() * 255).astype("uint8")
                arr = JArray(JByte)(norm8.ravel().tolist())
                iproc = self.ByteProcessor(w, h, arr, None)

            tile_images[idx] = iproc

        # 3. build the TileGrid and inject tiles
        grid = self.TileGrid(num_rows, num_cols)
        for idx, iproc in tile_images.items():
            r, c = divmod(idx, num_cols)
            tile = self.ImageTileClass(self.File("in_memory"), r, c)
            self._pixels.set(tile, iproc)
            self._width.setInt(tile, iproc.getWidth())
            self._height.setInt(tile, iproc.getHeight())
            self._loaded.setBoolean(tile, True)
            grid.setTile(r, c, tile)

        # 4. run MIST headless
        self.MISTMain.runHeadless = True
        if not self.MISTMain.runStitching(jp, grid):
            raise RuntimeError("MIST stitching failed")

        # 5. extract translation results
        results = []
        for idx in sorted(tile_images):
            r, c = divmod(idx, num_cols)
            t = grid.getTile(r, c)
            dx = getattr(t, "getXTranslation", lambda: 0)()
            dy = getattr(t, "getYTranslation", lambda: 0)()
            results.append({"tile_index": idx, "dx": dx, "dy": dy})

        return pd.DataFrame(results)
