from typing import Dict
import dask.array as da
import napari


class VolumeRenderer:
    """
    Renders 3D volume stacks in Napari.
    """
    def render_volume(self, viewer: napari.Viewer, pyramid: Dict[int, da.Array]) -> None:
        """
        Render the lowest-resolution 3D volume from the pyramid.

        Parameters
        ----------
        viewer : napari.Viewer
            Napari viewer instance.
        pyramid : Dict[int, dask.array.Array]
            Mapping from downsampling factor to 3D volume arrays.
        """
        # TODO: replace with NVIDIA IndeX integration
        # Select the lowest-resolution level (largest downsampling factor)
        level = max(pyramid.keys())
        volume = pyramid[level]
        # Add as an image layer for now (stubbed 3D stack display)
        viewer.add_image(
            volume,
            name=f"volume_level_{level}",
            multiscale=False,
            scale=[level] * volume.ndim,
        )
