import sys
import json
import numpy as np
import tifffile
from pathlib import Path
import pytest

def test_main_runs_and_creates_viewer(monkeypatch, tmp_path):
    # Prepare dummy data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create acquisition parameters.json
    import json
    params = {"sensor_pixel_size_um": 0.752}
    (data_dir / "acquisition parameters.json").write_text(json.dumps(params))

    # ✅ Write a valid stub TIFF
    arr = np.zeros((2048, 2048), dtype=np.uint8)
    tiff_path = data_dir / "manual_0_0_stub.tiff"
    tifffile.imwrite(tiff_path, arr)

    # Track that ViewerController.run is called
    called = {}
    def dummy_run(self, root):
        called['root'] = root

    # Monkey-patch ViewerController
    import rtviewer.main as m
    monkeypatch.setattr(
        m, 'ViewerController',
        lambda ds, cache, pyramid, adapter, renderer: type('C', (), {'run': dummy_run})()
    )

    # Simulate CLI args
    monkeypatch.setattr(sys, 'argv', ['rtviewer', '--dir', str(data_dir)])

    # Run main
    from rtviewer import main
    main.main()

    # Verify run() was called with correct root
    assert 'root' in called
    assert called['root'] == data_dir
