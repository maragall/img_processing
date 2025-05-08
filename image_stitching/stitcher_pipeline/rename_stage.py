from __future__ import annotations
import pandas as pd
from pathlib import Path
import os
from stitcher_pipeline.utils import iter_tiffs
from stitcher_pipeline.constants import FOV_SUFFIX_RE

def rename_stage(tile_dir: Path) -> None:
    csv_path = tile_dir / "coordinates.csv"
    df = pd.read_csv(csv_path)
    renamed = 0

    for p in iter_tiffs(tile_dir):
        fov, suffix = _split(p.name)
        fov_zp = f"{fov:03d}"
        new_name = f"manual_{fov_zp}_0_{suffix}"
        if p.name != new_name:
            os.rename(p, p.with_name(new_name))
            renamed += 1

    # pad FOV column in CSV
    df["fov"] = df["fov"].apply(lambda x: int(x))  # ensure int
    df.to_csv(csv_path, index=False)
    print(f"[rename] updated {renamed} TIFFs and rewrote coordinates.csv")

def _split(filename: str) -> tuple[int, str]:
    import re
    m = re.match(FOV_SUFFIX_RE, filename)
    if not m:
        raise ValueError(f"unrecognised filename: {filename}")
    return int(m.group(1)), m.group(2)
