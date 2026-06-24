"""Typed containers returned by the core pipeline functions.

The dataclasses in this module make it easier to pass structured analysis state
between notebook-style script cells without leaking implementation details into
project-specific user scripts.

author: Fabrizio Musacchio
date: May/June 2026
"""
# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# %% SCHEMAS
@dataclass(slots=True)
class ResultsPaths:
    """Collect all output locations for one input dataset."""

    source_path: Path
    results_dir: Path
    roi_mask_path: Path
    detailed_csv_path: Path
    excel_path: Path
    cell_mask_path: Path
    marker_mask_path: Path
    positive_cell_mask_path: Path
    optional_region_mask_path: Path


@dataclass(slots=True)
class SingleChannelResultsPaths:
    """Collect all output locations for one single-channel dataset run."""

    source_path: Path
    results_dir: Path
    roi_mask_path: Path
    object_csv_path: Path
    excel_path: Path
    mask_path: Path


@dataclass(slots=True)
class LoadedImageChannels:
    """Store the loaded analysis channels and related metadata.

    The stored image arrays may either represent the original ``ZYX`` channels
    loaded from disk or a derived analysis view, for example a z-projection
    prepared before ROI drawing and segmentation.
    """

    source_path: Path
    paths: ResultsPaths
    voxel_scale_zyx: tuple[float, float, float]
    cell_image: np.ndarray
    marker_image: np.ndarray
    optional_region_image: np.ndarray | None
    raw_shape_tzcyx: tuple[int, ...]
    raw_z_size: int
    is_3d: bool
    metadata: Any
    analysis_z_bounds: tuple[int, int] | None = None
    z_projection_method: str | None = None


@dataclass(slots=True)
class LoadedSingleChannelImage:
    """Store one loaded analysis channel and related metadata.

    The stored image array may represent either the original ``ZYX`` channel
    loaded from disk or a derived analysis view such as a z projection.
    """

    source_path: Path
    paths: SingleChannelResultsPaths
    voxel_scale_zyx: tuple[float, float, float]
    image: np.ndarray
    raw_shape_tzcyx: tuple[int, ...]
    raw_z_size: int
    is_3d: bool
    metadata: Any
    analysis_z_bounds: tuple[int, int] | None = None
    z_projection_method: str | None = None


@dataclass(slots=True)
class OptionalRegionSegmentationResult:
    """Hold the results of the optional third-channel segmentation."""

    mask: np.ndarray
    labels: np.ndarray
    threshold: float
    corrected_image: np.ndarray


@dataclass(slots=True)
class ColocalizationTables:
    """Bundle all exported multi-channel colocalization result tables.

    Attributes
    ----------
    detailed:
        Per-overlap table containing one row per cell-marker overlap event.
    summary:
        Main per-cell colocalization table for the cell channel, augmented with
        cell-channel morphology metrics.
    overview:
        Existing per-ROI colocalization overview table. This is still exposed
        under the historic attribute name ``overview`` for backward
        compatibility, even though the Excel export sheet is named
        ``roi_coloc_overview``.
    marker_properties:
        Per-object morphology table for the segmented marker channel.
    third_channel_properties:
        Optional per-object morphology table for the segmented third channel.
    roi_cell_summary:
        Per-ROI means of the cell-channel morphology metrics.
    roi_marker_summary:
        Per-ROI means of the marker-channel morphology metrics.
    roi_third_channel_summary:
        Optional per-ROI means of the third-channel morphology metrics.
    """

    detailed: pd.DataFrame
    summary: pd.DataFrame
    overview: pd.DataFrame
    marker_properties: pd.DataFrame | None = None
    third_channel_properties: pd.DataFrame | None = None
    roi_cell_summary: pd.DataFrame | None = None
    roi_marker_summary: pd.DataFrame | None = None
    roi_third_channel_summary: pd.DataFrame | None = None


@dataclass(slots=True)
class SingleChannelTables:
    """Bundle tables produced by one-channel analyses.

    Attributes
    ----------
    objects:
        Main per-object summary table containing biologically meaningful size
        and shape descriptors such as area, volume, surface area, roundness,
        or sphericity, depending on whether the analysis view is effectively
        2D or 3D.
    voxel_plausibility:
        Technical cross-check table comparing direct voxel counts against
        ``regionprops``-derived voxel counts for each object.
    overview:
        Per-ROI overview table containing object counts, occupancy metrics,
        and per-ROI means of the object-summary shape descriptors.
    """

    objects: pd.DataFrame
    voxel_plausibility: pd.DataFrame
    overview: pd.DataFrame


@dataclass(slots=True)
class CellposeRefinementRoiCache:
    """Store the per-ROI network outputs needed for fast threshold refinement.

    The expensive neural network forward pass is performed only once. The
    stored ``dP`` and ``cellprob`` arrays can then be reused to recompute masks
    with different ``cellprob_threshold`` and ``flow_threshold`` values without
    rerunning the network on the raw image.
    """

    roi_id: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    roi_mask_crop_2d: np.ndarray
    shape_for_masks: tuple[int, int, int]
    dP: np.ndarray
    cellprob: np.ndarray
    do_3d: bool
    niter: int
    min_size: int
    max_size_fraction: float
    flow_threshold: float
    cellprob_threshold: float


@dataclass(slots=True)
class CellposeChannelRefinementContext:
    """Bundle all ROI caches and the model reference for one image channel."""

    model: Any
    model_name_or_path: str
    roi_caches: list[CellposeRefinementRoiCache]


@dataclass(slots=True)
class ColocalizationRunResult:
    """Bundle label masks and tables generated by one completed analysis run.

    Attributes
    ----------
    cell_masks, marker_masks, positive_cell_masks:
        Full-stack label images for the primary cell channel, the marker
        channel, and the subset of marker-positive cells.
    tables:
        Detailed, per-cell summary, and per-ROI overview tables.
    optional_region_masks:
        Optional third-channel label image when such a channel was segmented.
    analysis_z_bounds:
        Optional global z interval actually used for internal segmentation and
        quantification. Arrays remain full-stack sized even when this is set.
    cell_refinement_context, marker_refinement_context,
    optional_region_refinement_context:
        Optional cached Cellpose network outputs that allow later threshold-only
        refinement without rerunning the neural network forward pass.
    """

    cell_masks: np.ndarray
    marker_masks: np.ndarray
    positive_cell_masks: np.ndarray
    tables: ColocalizationTables
    optional_region_masks: np.ndarray | None = None
    analysis_z_bounds: tuple[int, int] | None = None
    cell_refinement_context: CellposeChannelRefinementContext | None = None
    marker_refinement_context: CellposeChannelRefinementContext | None = None
    optional_region_refinement_context: CellposeChannelRefinementContext | None = None


@dataclass(slots=True)
class SingleChannelRunResult:
    """Bundle masks and tables generated by one single-channel analysis run.

    Attributes
    ----------
    masks:
        Full-stack label image for the segmented objects of the analyzed
        channel.
    tables:
        Per-object and per-ROI summary tables.
    analysis_z_bounds:
        Optional global z interval used internally for segmentation and
        quantification. Arrays remain full-stack sized even when this is set.
    refinement_context:
        Optional cached Cellpose network outputs that allow later threshold-only
        refinement without rerunning the neural network forward pass.
    """

    masks: np.ndarray
    tables: SingleChannelTables
    analysis_z_bounds: tuple[int, int] | None = None
    refinement_context: CellposeChannelRefinementContext | None = None
# %% END
