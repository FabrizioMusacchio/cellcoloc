"""Reusable core pipeline for interactive Cellpose-based colocalization.

The package is intentionally structured so that project-specific user scripts
can stay small: they define settings, call the imported pipeline functions cell
by cell, and write all outputs to a standardized ``results`` directory located
next to the source dataset.
"""

from __future__ import annotations

from .runtime import get_runtime_cache_root, prepare_runtime_environment

prepare_runtime_environment()

from .analysis import build_positive_cell_mask, run_roi_cellpose_colocalization
from .config import (
    CellposeModelConfig,
    ChannelConfig,
    ColocalizationConfig,
    DisplayNames,
    OptionalRegionSegmentationConfig,
    RuntimeConfig,
)
from .io import build_results_paths, export_analysis_outputs, load_analysis_images, load_roi_labels, save_roi_labels
from .roi import (
    create_full_image_roi_labels,
    create_roi_drawing_viewer,
    get_bbox_2d,
    get_roi_label_points,
    rasterize_shapes_to_labelmask,
    save_roi_labels_from_shapes,
)
from .schemas import (
    ColocalizationRunResult,
    ColocalizationTables,
    LoadedImageChannels,
    OptionalRegionSegmentationResult,
    ResultsPaths,
)
from .segmentation import (
    create_cellpose_model,
    create_cellpose_models_for_channels,
    evaluate_cellpose_model,
    filter_labels_by_size,
    get_cellpose_major_version,
    get_available_cellpose_model_names,
    relabel_with_offset,
    segment_optional_region,
)
from .visualization import show_analysis_results, show_optional_region_segmentation

__all__ = [
    "CellposeModelConfig",
    "ChannelConfig",
    "ColocalizationConfig",
    "DisplayNames",
    "OptionalRegionSegmentationConfig",
    "RuntimeConfig",
    "ResultsPaths",
    "LoadedImageChannels",
    "OptionalRegionSegmentationResult",
    "ColocalizationTables",
    "ColocalizationRunResult",
    "build_results_paths",
    "load_analysis_images",
    "save_roi_labels",
    "load_roi_labels",
    "export_analysis_outputs",
    "prepare_runtime_environment",
    "get_runtime_cache_root",
    "create_full_image_roi_labels",
    "rasterize_shapes_to_labelmask",
    "create_roi_drawing_viewer",
    "save_roi_labels_from_shapes",
    "get_bbox_2d",
    "get_roi_label_points",
    "create_cellpose_model",
    "create_cellpose_models_for_channels",
    "evaluate_cellpose_model",
    "relabel_with_offset",
    "filter_labels_by_size",
    "get_cellpose_major_version",
    "get_available_cellpose_model_names",
    "segment_optional_region",
    "run_roi_cellpose_colocalization",
    "build_positive_cell_mask",
    "show_optional_region_segmentation",
    "show_analysis_results",
]
