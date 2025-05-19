# Overview

- **Data viewer without running the data through the stitcher**
  - 2–3 layer image pyramid??? Maybe just a single layer? Or generate the pyramid on the fly?
    - **components:**
      - Registration (can be on the fly; or use existing code)
      - “Stitch” on the fly for display (4, 8, 16 FOVs at a time)
    - **UI:**
      - Constraints (e.g. do not allow zoom-out beyond certain demagnification)
      - “Map view” (window in top-left corner showing a “red dot” of where the user is located within the thumbnail of the sample)
  - Handling changing _z_ (should be easy)
  - 3D view of the current FOV or tiles (will require loading the entire stack into RAM – downsample to fit RAM).  
    _Stretch goal:_ use NVIDIA INdex
