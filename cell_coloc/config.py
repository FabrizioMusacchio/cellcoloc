"""Configuration objects for the reusable cell colocalization pipeline.

This module groups all user-adjustable settings into small dataclasses so that
project-specific scripts only need to define parameters and then call the core
pipeline functions exposed by :mod:`cell_coloc`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


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
    segmentation_method:
        Segmentation backend used for this channel. ``"cellpose"`` keeps the
        existing neural-network workflow. ``"otsu"``, ``"li"``, and
        ``"percentile"`` use intensity thresholding followed by connected
        component labeling.
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
    flow3d_smooth:
        Optional Gaussian smoothing strength forwarded to Cellpose for 3D flow
        fields. This setting only has an effect for true 3D runs and is
        ignored for 2D data. Keep the default ``0`` to disable smoothing.
    prefilter:
        Optional image prefilter applied before Cellpose. Supported values are
        ``None``, ``"gaussian"``, ``"laplacian_of_gaussian"`` (alias
        ``"log"``), and ``"median"``.
    prefilter_sigma_xy:
        Gaussian prefilter sigma in the in-plane directions. Used when
        ``prefilter="gaussian"``.
    prefilter_sigma_z:
        Gaussian prefilter sigma along z. If ``None``, the pipeline reuses
        ``prefilter_sigma_xy``. Only relevant for 3D data.
    prefilter_median_size_xy:
        Median prefilter kernel size in the in-plane directions. Used when
        ``prefilter="median"``.
    prefilter_median_size_z:
        Median prefilter kernel size along z. If ``None``, the pipeline reuses
        ``prefilter_median_size_xy``. Only relevant for 3D data.
    threshold_percentile:
        Percentile used when ``segmentation_method="percentile"``.
    threshold_background_sigma:
        Optional Gaussian sigma used for background subtraction before
        threshold-based segmentation.
    threshold_min_object_voxels:
        Minimum object size kept after threshold-based segmentation.
    threshold_min_hole_voxels:
        Minimum hole size filled after threshold-based segmentation.
    threshold_apply_closing:
        Whether a small binary closing step should be applied before threshold
        cleanup.
    postfilters:
        Optional post-segmentation filters applied to the resulting masks.
        Supported values are ``None``, ``"min_intensity"``,
        ``"local_contrast"``, ``"bright_pixel_support"``, or a list combining
        them in the requested execution order.
    min_intensity_measure:
        Statistic used by the ``"min_intensity"`` postfilter. Supported values
        are ``"mean"``, ``"median"``, and ``"max"``.
    min_intensity_threshold:
        Intensity threshold used by the ``"min_intensity"`` postfilter.
    local_contrast_k:
        Contrast multiplier used by the ``"local_contrast"`` postfilter in the
        criterion ``object_median > background_median + k * background_mad``.
    local_contrast_shell_inner_radius:
        Inner dilation radius, in pixels or voxels, used to construct the
        local shell for the ``"local_contrast"`` postfilter.
    local_contrast_shell_outer_radius:
        Outer dilation radius, in pixels or voxels, used to construct the
        local shell for the ``"local_contrast"`` postfilter.
    bright_pixel_measure:
        Statistic used by the ``"bright_pixel_support"`` postfilter.
        ``"count"`` requires at least a minimum number of bright pixels within
        the mask, while ``"fraction"`` requires a minimum fraction of bright
        pixels relative to the object size.
    bright_pixel_threshold:
        Intensity threshold above which a pixel or voxel counts as bright for
        the ``"bright_pixel_support"`` postfilter.
    bright_pixel_min_count:
        Minimum number of bright pixels or voxels required when
        ``bright_pixel_measure="count"``.
    bright_pixel_min_fraction:
        Minimum fraction of bright pixels or voxels required when
        ``bright_pixel_measure="fraction"``.
    """

    model_name_or_path: str
    segmentation_method: str = "cellpose"
    diameter: float | None = None
    do_3d: bool | None = None
    z_axis: int = 0
    channel_axis: int | None = None
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    anisotropy: bool | float = False
    flow3d_smooth: int = 0
    prefilter: str | None = None
    prefilter_sigma_xy: float = 1.0
    prefilter_sigma_z: float | None = None
    prefilter_median_size_xy: int = 3
    prefilter_median_size_z: int | None = None
    threshold_percentile: float = 98.0
    threshold_background_sigma: float | None = None
    threshold_min_object_voxels: int = 10
    threshold_min_hole_voxels: int = 10
    threshold_apply_closing: bool = True
    postfilters: str | Sequence[str] | None = None
    min_intensity_measure: str = "mean"
    min_intensity_threshold: float | None = None
    local_contrast_k: float = 1.0
    local_contrast_shell_inner_radius: int = 1
    local_contrast_shell_outer_radius: int = 4
    bright_pixel_measure: str = "count"
    bright_pixel_threshold: float | None = None
    bright_pixel_min_count: int | None = None
    bright_pixel_min_fraction: float | None = None


@dataclass(slots=True)
class ColocalizationConfig:
    """Thresholds controlling how cell-to-marker overlap is interpreted.

    Attributes
    ----------
    min_cell_voxels:
        Minimum size required for cell labels after segmentation cleanup.
    overlap_fraction_threshold:
        Minimum fraction of the cell mask that must overlap a marker object for
        the cell to be counted as positive.
    min_overlap_voxels:
        Minimum absolute overlap size required for positivity.
    evaluate_optional_region_cell_positivity:
        When ``True`` and an optional third channel has been segmented, the
        pipeline additionally evaluates which cells overlap that third-channel
        segmentation. This yields separate optional-region positivity columns
        as well as a combined marker-and-optional-region double-positive flag.
    """

    min_cell_voxels: int = 200
    overlap_fraction_threshold: float = 0.02
    min_overlap_voxels: int = 20
    evaluate_optional_region_cell_positivity: bool = False


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
    """Execution-time settings that control interactivity and test crops.

    Attributes
    ----------
    draw_rois:
        Whether the interactive ROI-drawing step should be used.
    process_rois:
        Whether ROI-wise processing should be executed.
    open_results:
        Whether napari result viewers should be opened.
    use_gpu:
        Whether GPU execution should be requested for Cellpose.
    crop_for_testing:
        Optional test crop in ``(Z, Y, X)`` order.
    image_loading_mode:
        Controls how OMIO loads the raw microscopy file. ``"memory"`` loads
        the image eagerly in memory and forwards ``zarr_store=None`` to
        :func:`omio.imread`. ``"memap"`` uses ``zarr_store="disk"`` with
        ``reuse_disk_cache=True`` so large stacks can be backed by disk cache
        instead of being fully materialized up front.
    """

    draw_rois: bool = True
    process_rois: bool = True
    open_results: bool = True
    use_gpu: bool = True
    crop_for_testing: tuple[slice, slice, slice] | None = None
    image_loading_mode: str = "memory"
