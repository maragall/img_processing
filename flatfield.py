#!/usr/bin/env python3
import logging
import random
from pathlib import Path

import numpy as np
from basicpy import BaSiC
from skimage.io import imread
import tifffile

# ─── CONFIG: hard-coded list of datasets ──────────────────────────────────────
INPUT_DIRS = [
    Path("/home/cephla/Downloads/widefield_2025-03-25_15-34-46.186779/0"),
    Path("/home/cephla/Downloads/confocal_100_ms_2025-03-25_15-40-46.833874/0"),
    Path("/home/cephla/Downloads/widefield_2025-03-25_15-37-00.025884/0"),
]
MAX_FLATFIELD_IMAGES = 48
CHANNELS = ["405", "488", "561", "638"]
SUFFIX_TMPL = "Fluorescence_{ch}_nm_Ex"
# ───────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    for input_dir in INPUT_DIRS:
        logging.info(f"=== Processing {input_dir} ===")

        # 1) compute flatfields
        flatfields = {}
        for ch in CHANNELS:
            suffix = SUFFIX_TMPL.format(ch=ch)
            tiffs = sorted(input_dir.glob(f"*_{suffix}.tiff"))
            if not tiffs:
                logging.warning(f"No TIFFs for channel {ch} in {input_dir}")
                continue

            random.shuffle(tiffs)
            sample = tiffs[: min(MAX_FLATFIELD_IMAGES, len(tiffs))]
            if len(sample) < len(tiffs):
                logging.warning(f"Using {len(sample)}/{len(tiffs)} images for {ch}")

            # load sample as float for BaSiC, but remember original dtype
            imgs = []
            dtypes = []
            for p in sample:
                raw = imread(str(p))
                dtypes.append(raw.dtype)
                imgs.append(raw.astype(np.float32))
            arr = np.stack(imgs, axis=0)
            if arr.ndim not in (3, 4):
                logging.error(f"Bad dims {arr.shape} for {ch}, skipping")
                continue

            logging.info(f"Fitting BaSiC flatfield for channel {ch} (shape={arr.shape})")
            basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
            basic.fit(arr)
            flatfields[ch] = basic.flatfield

        # 2) apply & overwrite every TIFF in each channel
        for ch, ff in flatfields.items():
            suffix = SUFFIX_TMPL.format(ch=ch)
            mean_ff = ff.mean()
            for path in sorted(input_dir.glob(f"*_{suffix}.tiff")):
                raw = imread(str(path))
                orig_dtype = raw.dtype

                # compute correction in float
                corr = (raw.astype(np.float32) / ff) * mean_ff

                # clip and cast back to original integer dtype
                info = np.iinfo(orig_dtype)
                corr = np.clip(corr, info.min, info.max).astype(orig_dtype)

                tifffile.imwrite(str(path), corr)
                logging.info(f"  corrected & overwrote {path.name}")

if __name__ == "__main__":
    main()
