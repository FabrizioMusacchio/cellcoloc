"""Reusable core pipeline for interactive Cellpose-based colocalization.

The package is intentionally structured so that project-specific user scripts
can stay small: they define settings, call the imported pipeline functions cell
by cell, and write all outputs to a standardized ``results`` directory located
next to the source dataset.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

_CACHE_ROOT = Path(tempfile.gettempdir()) / "cell_coloc_runtime_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "numba").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg_cache").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "napari").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))
os.environ.setdefault("NAPARI_CONFIG", str(_CACHE_ROOT / "napari" / "settings.yaml"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg_cache"))

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
from .roi import create_roi_drawing_viewer, get_bbox_2d, get_roi_label_points, rasterize_shapes_to_labelmask, save_roi_labels_from_shapes
from .schemas import (
    ColocalizationRunResult,
    ColocalizationTables,
    LoadedImageChannels,
    OptionalRegionSegmentationResult,
    ResultsPaths,
)
from .segmentation import (
    create_cellpose_model,
    evaluate_cellpose_model,
    filter_labels_by_size,
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
    "rasterize_shapes_to_labelmask",
    "create_roi_drawing_viewer",
    "save_roi_labels_from_shapes",
    "get_bbox_2d",
    "get_roi_label_points",
    "create_cellpose_model",
    "evaluate_cellpose_model",
    "relabel_with_offset",
    "filter_labels_by_size",
    "get_available_cellpose_model_names",
    "segment_optional_region",
    "run_roi_cellpose_colocalization",
    "build_positive_cell_mask",
    "show_optional_region_segmentation",
    "show_analysis_results",
]
