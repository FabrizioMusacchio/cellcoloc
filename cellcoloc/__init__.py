"""Reusable core pipeline for interactive segmentation-based colocalization.

The package is intentionally structured so that project-specific user scripts
can stay small: they define settings, call the imported pipeline functions cell
by cell, and write all outputs to a standardized ``results`` directory located
next to the source dataset.

author: Fabrizio Musacchio
date: May/June 2026
"""

from __future__ import annotations

from .runtime import get_runtime_cache_root, prepare_runtime_environment

prepare_runtime_environment()

from .analysis import (
    analyze_existing_masks,
    build_positive_cell_mask,
    prepare_loaded_images_for_analysis,
    refine_run_result_from_cellpose_cache,
    run_roi_cellpose_colocalization,
)
from .config import (
    CellposeModelConfig,
    ChannelConfig,
    ColocalizationConfig,
    DisplayNames,
    OptionalRegionSegmentationConfig,
    RuntimeConfig,
    SingleChannelAnalysisConfig,
    SingleChannelConfig,
    SingleChannelDisplayNames,
)
from .io import (
    build_results_paths,
    export_analysis_outputs,
    load_analysis_images,
    load_roi_labels,
    save_roi_labels,
    try_load_roi_labels,
)
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
    LoadedSingleChannelImage,
    SingleChannelResultsPaths,
    SingleChannelRunResult,
    SingleChannelTables,
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
from .visualization import extract_label_masks_from_viewer
from .single_channel import (
    analyze_existing_single_channel_masks,
    build_single_channel_results_paths,
    create_single_channel_roi_drawing_viewer,
    export_single_channel_outputs,
    extract_single_channel_masks_from_viewer,
    load_single_channel_image,
    prepare_loaded_single_channel_image_for_analysis,
    refine_single_channel_run_result_from_cellpose_cache,
    run_roi_single_channel_segmentation,
    show_single_channel_results,
    try_load_single_channel_roi_labels,
)

__all__ = [
    "CellposeModelConfig",
    "ChannelConfig",
    "ColocalizationConfig",
    "DisplayNames",
    "OptionalRegionSegmentationConfig",
    "RuntimeConfig",
    "SingleChannelConfig",
    "SingleChannelDisplayNames",
    "SingleChannelAnalysisConfig",
    "ResultsPaths",
    "LoadedImageChannels",
    "SingleChannelResultsPaths",
    "LoadedSingleChannelImage",
    "OptionalRegionSegmentationResult",
    "ColocalizationTables",
    "ColocalizationRunResult",
    "SingleChannelTables",
    "SingleChannelRunResult",
    "build_results_paths",
    "build_single_channel_results_paths",
    "load_analysis_images",
    "load_single_channel_image",
    "save_roi_labels",
    "load_roi_labels",
    "try_load_roi_labels",
    "try_load_single_channel_roi_labels",
    "export_analysis_outputs",
    "export_single_channel_outputs",
    "analyze_existing_masks",
    "analyze_existing_single_channel_masks",
    "prepare_loaded_images_for_analysis",
    "prepare_loaded_single_channel_image_for_analysis",
    "prepare_runtime_environment",
    "get_runtime_cache_root",
    "create_full_image_roi_labels",
    "rasterize_shapes_to_labelmask",
    "create_roi_drawing_viewer",
    "create_single_channel_roi_drawing_viewer",
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
    "run_roi_single_channel_segmentation",
    "refine_run_result_from_cellpose_cache",
    "refine_single_channel_run_result_from_cellpose_cache",
    "build_positive_cell_mask",
    "extract_label_masks_from_viewer",
    "extract_single_channel_masks_from_viewer",
    "show_optional_region_segmentation",
    "show_analysis_results",
    "show_single_channel_results",
]
