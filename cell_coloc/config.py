"""Configuration objects for the reusable cell colocalization pipeline.

This module groups all user-adjustable settings into small dataclasses so that
project-specific scripts only need to define parameters and then call the core
pipeline functions exposed by :mod:`cell_coloc`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChannelConfig:
    """Describe which channels should be used for the analysis.

    Attributes
    ----------
    cell_channel:
        Zero-based channel index containing the larger cellular compartment that
        should be segmented with Cellpose, for example cytoskeleton or soma.
    marker_channel:
        Zero-based channel index containing the marker-positive structure that
        determines the colocalization class, for example nuclei or DAPI.
    optional_region_channel:
        Optional zero-based channel index for a third channel that is
        thresholded to estimate a region coverage metric, for example tumor
        infiltration. Set to ``None`` to disable the optional third-channel
        analysis entirely.
    """

    cell_channel: int
    marker_channel: int
    optional_region_channel: int | None = None


@dataclass(slots=True)
class DisplayNames:
    """Human-readable names used in napari layers and user-facing messages."""

    cell: str = "Cells"
    marker: str = "Marker"
    optional_region: str = "Optional region"
    positive_cells: str = "Marker-positive cells"


@dataclass(slots=True)
class CellposeModelConfig:
    """Collect Cellpose settings for one segmentation target.

    Attributes
    ----------
    diameter:
        Diameter parameter passed to :meth:`cellpose.models.CellposeModel.eval`.
        For Cellpose 4 and newer this can be set to ``None`` to let Cellpose
        choose its default behavior without an explicit diameter.
    model_name_or_path:
        Either a built-in Cellpose model identifier such as ``"cyto3"`` or
        ``"nuclei"``, or a filesystem path pointing to a custom trained model.
    do_3d:
        Whether Cellpose should run in 3D mode. If set to ``None``, the
        pipeline auto-detects 2D versus 3D from the loaded image z-size.
    z_axis:
        Array axis representing the z dimension for Cellpose.
    channel_axis:
        Channel axis passed to Cellpose. Keep this as ``None`` for single
        channel volumes.
    cellprob_threshold:
        Cellpose pixel probability threshold used during mask generation. This
        is only forwarded explicitly for Cellpose 4 and newer.
    flow_threshold:
        Cellpose flow error threshold used during mask generation. This is only
        forwarded explicitly for Cellpose 4 and newer.
    anisotropy:
        Controls whether a 3D Cellpose run should use anisotropy correction.
        Set this to ``False`` to disable anisotropy handling entirely, to
        ``True`` to let the pipeline derive an anisotropy factor from the
        configured voxel size, or to a numeric value to force a specific
        Cellpose anisotropy factor. The value is ignored for 2D runs.
    """

    model_name_or_path: str
    diameter: float | None = None
    do_3d: bool | None = None
    z_axis: int = 0
    channel_axis: int | None = None
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    anisotropy: bool | float = False


@dataclass(slots=True)
class ColocalizationConfig:
    """Thresholds controlling how cell-to-marker overlap is interpreted."""

    min_cell_voxels: int = 200
    overlap_fraction_threshold: float = 0.02
    min_overlap_voxels: int = 20


@dataclass(slots=True)
class OptionalRegionSegmentationConfig:
    """Settings for the optional threshold-based third-channel analysis.

    The default pipeline is a two-channel colocalization problem. When
    ``enabled`` is set to ``True`` and a third channel is configured, the
    pipeline additionally segments a thresholded region mask and reports its
    area and volume fraction inside each ROI.
    """

    enabled: bool = False
    method: str = "otsu"
    percentile: float = 98.0
    gaussian_sigma: float | None = 1.0
    background_sigma: float | None = None
    min_object_voxels: int = 10
    min_hole_voxels: int = 10
    apply_closing: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    """Execution-time settings that control interactivity and test crops."""

    draw_rois: bool = True
    process_rois: bool = True
    open_results: bool = True
    use_gpu: bool = True
    crop_for_testing: tuple[slice, slice, slice] | None = None
