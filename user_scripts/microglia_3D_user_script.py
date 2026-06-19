"""Interactive user script for microglia colocalization analysis.

This script is configured for the datasets stored in
``example_data/microglia_2D``. Despite the folder name, the core pipeline now
auto-detects whether each file is truly 2D or 3D by inspecting the OME/TZCYX
z dimension after loading with :mod:`omio`.

The biological colocalization task in this script is:

- channel 0: ``Cx3cr1-tdTomato`` microglia reporter signal,
- channel 1: ``Iba1`` staining,
- channel 2: ``DAPI`` only for orientation and optional visualization.

The quantitative colocalization is performed between channel 0 and channel 1.
ROIs can be drawn interactively in napari or, if desired, the whole field of
view can be analyzed as one single ROI.
"""
# %% IMPORTS AND LOCAL PACKAGE BOOTSTRAP
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cell_coloc import (
    CellposeModelConfig,
    ChannelConfig,
    ColocalizationConfig,
    DisplayNames,
    RuntimeConfig,
    create_full_image_roi_labels,
    create_roi_drawing_viewer,
    export_analysis_outputs,
    load_analysis_images,
    load_roi_labels,
    refine_run_result_from_cellpose_cache,
    run_roi_cellpose_colocalization,
    save_roi_labels_from_shapes,
    show_analysis_results,
    try_load_roi_labels,
)

import napari
import numpy as np
# %% PROJECT SETTINGS
DATA_DIR = PROJECT_ROOT / "example_data" / "microglia_3D"
DATA_PATHS = sorted(DATA_DIR.glob("*.czi"))

CHANNEL_CONFIG = ChannelConfig(
    cell_channel=0,
    marker_channel=1,
    optional_region_channel=2,
)

DISPLAY_NAMES = DisplayNames(
    cell="Cx3cr1-tdTomato",
    marker="Iba1",
    optional_region="DAPI",
    positive_cells="tdTomato + Iba1 positive masks")

# Placeholder values. Replace with the true metadata-derived pixel sizes when
# available for quantitative micrometer-scale interpretation.
VOXEL_SCALE_ZYX = (1.0, 0.6239258, 0.6239258)

CELL_MODEL_CONFIG = CellposeModelConfig(
    # diameter=15,
    model_name_or_path="cpsam",  # cpsam for Cellpose 4, cyto3 for Cellpose 3
    anisotropy=True,
    flow3d_smooth=3,  # Gaussian smoothing for 3D flow fields; int, default: 0, range: 0-10
    prefilter="gaussian",  # available options: "gaussian", "laplacian_of_gaussian"/"log", "median", None
        prefilter_sigma_xy=0.8,
        prefilter_sigma_z=0.0,
        prefilter_median_size_xy=3,
        prefilter_median_size_z=3,
    postfilters=None,  # available options: "min_intensity", "local_contrast", "bright_pixel_support", None
        min_intensity_measure="mean",
        min_intensity_threshold=None,
        local_contrast_k=1.0,
        local_contrast_shell_inner_radius=1,
        local_contrast_shell_outer_radius=4,
        bright_pixel_measure="count",
        bright_pixel_threshold=None,
        bright_pixel_min_count=None,
        bright_pixel_min_fraction=None,
    cellprob_threshold=1.5,
    flow_threshold=0.4)

MARKER_MODEL_CONFIG = CellposeModelConfig(
    # diameter=15,
    model_name_or_path="cpsam",  # cpsam for Cellpose 4, cyto3 for Cellpose 3
    anisotropy=True,
    flow3d_smooth=3,
    prefilter="laplacian_of_gaussian",
        prefilter_sigma_xy=2.0,
        prefilter_sigma_z=0.1,
        prefilter_median_size_xy=3,
        prefilter_median_size_z=3,
    postfilters=None,
        min_intensity_measure="mean",
        min_intensity_threshold=None,
        local_contrast_k=1.0,
        local_contrast_shell_inner_radius=1,
        local_contrast_shell_outer_radius=4,
        bright_pixel_measure="count",
        bright_pixel_threshold=None,
        bright_pixel_min_count=None,
        bright_pixel_min_fraction=None,
    cellprob_threshold=0,
    flow_threshold=0.4)

COLOCALIZATION_CONFIG = ColocalizationConfig(
    min_cell_voxels=50,
    overlap_fraction_threshold=0.02,
    min_overlap_voxels=10,
)

RUNTIME_CONFIG = RuntimeConfig(
    draw_rois=True,
    process_rois=True,
    open_results=True,
    use_gpu=True,
    crop_for_testing=None,
)

if not DATA_PATHS:
    raise FileNotFoundError(f"No CZI files found in:\n{DATA_DIR}")

print("Detected input files:")
for data_path in DATA_PATHS:
    print(f" - {data_path.name}")

SELECTED_FILE_NAME = DATA_PATHS[0].name
USE_FULL_IMAGE_AS_SINGLE_ROI = False
REUSE_EXISTING_ROI_MASK_IF_AVAILABLE = True
INITIAL_RESULT_LAYER_KEYS = [
    "cell_image",
    "marker_image",
    "optional_region_image",
    "rois",
    "roi_numbers",
    "cell_masks",
    "marker_masks",
    "positive_cells",
]
REFINEMENT_RESULT_LAYER_KEYS = [
    "cell_masks",
    "marker_masks",
    "positive_cells",
]

DATA_PATH = DATA_DIR / SELECTED_FILE_NAME
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Selected file does not exist:\n{DATA_PATH}")
# %% LOAD THE ANALYSIS CHANNELS
loaded_images = load_analysis_images(
    source_path=DATA_PATH,
    channel_config=CHANNEL_CONFIG,
    voxel_scale_zyx=VOXEL_SCALE_ZYX,
    crop_for_testing=RUNTIME_CONFIG.crop_for_testing)
print(f"Results directory:\n{loaded_images.paths.results_dir}")
existing_roi_labels = None
if REUSE_EXISTING_ROI_MASK_IF_AVAILABLE:
    existing_roi_labels = try_load_roi_labels(loaded_images.paths.roi_mask_path)
# %% DRAW ROIS INTERACTIVELY IN NAPARI
if USE_FULL_IMAGE_AS_SINGLE_ROI:
    print("Whole-image mode is enabled. ROI drawing is skipped.")
elif existing_roi_labels is not None:
    print("An existing ROI mask was found and will be reused. ROI drawing is skipped.")
elif RUNTIME_CONFIG.draw_rois:
    roi_viewer, shapes_layer = create_roi_drawing_viewer(
        loaded_images=loaded_images,
        display_names=DISPLAY_NAMES)
    print("Draw ROIs in napari and close the window. Then run the next cell.")
    napari.run()
else:
    print("ROI drawing is disabled. The next cell will load an existing ROI mask from disk.")
# %% SAVE THE DRAWN ROIS OR LOAD AN EXISTING ROI MASK
if USE_FULL_IMAGE_AS_SINGLE_ROI:
    roi_labels_2d = create_full_image_roi_labels(loaded_images.cell_image.shape[1:])
elif existing_roi_labels is not None:
    roi_labels_2d = existing_roi_labels
elif RUNTIME_CONFIG.draw_rois:
    roi_labels_2d = save_roi_labels_from_shapes(
        shapes_layer=shapes_layer,
        output_path=loaded_images.paths.roi_mask_path,
        image_shape_yx=loaded_images.cell_image.max(axis=0).shape,
        scale_yx=loaded_images.voxel_scale_zyx[1:])
else:
    roi_labels_2d = load_roi_labels(loaded_images.paths.roi_mask_path)

roi_ids = np.unique(roi_labels_2d)
roi_ids = roi_ids[roi_ids != 0]
print(f"ROI ids: {roi_ids}")
result_viewer = None
# %% RUN THE ROI-WISE CELLPOSE COLOCALIZATION
run_result = run_roi_cellpose_colocalization(
    loaded_images=loaded_images,
    roi_labels_2d=roi_labels_2d,
    cell_model_config=CELL_MODEL_CONFIG,
    marker_model_config=MARKER_MODEL_CONFIG,
    colocalization_config=COLOCALIZATION_CONFIG,
    runtime_config=RUNTIME_CONFIG,
    optional_region_result=None)
print(run_result.tables.overview)
# %% VISUALIZE THE RESULT IN NAPARI
if RUNTIME_CONFIG.open_results:
    result_viewer = None
    result_viewer = show_analysis_results(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        display_names=DISPLAY_NAMES,
        optional_region_result=None,
        viewer=result_viewer,
        layers_to_show=INITIAL_RESULT_LAYER_KEYS,
        replace_existing_layers=True,
        show_optional_region_image=True)

    print(f"Inspecting visualization for:\n{SELECTED_FILE_NAME}")
    napari.run()
# %% OPTIONALLY REFINE RESULTS AND VISUALIZE UPDATED RESULT IN NAPARI
REFINE_WITH_CACHED_CELLPOSE_OUTPUTS = True

REFINED_CELL_CELLPROB_THRESHOLD = CELL_MODEL_CONFIG.cellprob_threshold- 2.0 # - 0.9
REFINED_CELL_FLOW_THRESHOLD = CELL_MODEL_CONFIG.flow_threshold+0.1

REFINED_MARKER_CELLPROB_THRESHOLD = -3#4.5
REFINED_MARKER_FLOW_THRESHOLD = 0.4#0.8

REFINED_CELL_POSTFILTERS = ["min_intensity", "bright_pixel_support", "local_contrast"] # available options: "min_intensity", "local_contrast", "bright_pixel_support", None
REFINED_CELL_MIN_INTENSITY_MEASURE = "max"
REFINED_CELL_MIN_INTENSITY_THRESHOLD = 250
REFINED_CELL_LOCAL_CONTRAST_K = 1
REFINED_CELL_LOCAL_CONTRAST_SHELL_INNER_RADIUS = 1.0
REFINED_CELL_LOCAL_CONTRAST_SHELL_OUTER_RADIUS = 10
REFINED_CELL_BRIGHT_PIXEL_MEASURE = "fraction"
REFINED_CELL_BRIGHT_PIXEL_THRESHOLD = 250
REFINED_CELL_BRIGHT_PIXEL_MIN_COUNT = None
REFINED_CELL_BRIGHT_PIXEL_MIN_FRACTION = 0.08

#REFINED_MARKER_POSTFILTERS = ["min_intensity", "bright_pixel_support", "local_contrast"] #"local_contrast" # available options: "min_intensity", "local_contrast", "bright_pixel_support", None
REFINED_MARKER_POSTFILTERS = ["min_intensity", "bright_pixel_support", "local_contrast"]
REFINED_MARKER_MIN_INTENSITY_MEASURE = "max"
REFINED_MARKER_MIN_INTENSITY_THRESHOLD = 250
REFINED_MARKER_LOCAL_CONTRAST_K = 1
REFINED_MARKER_LOCAL_CONTRAST_SHELL_INNER_RADIUS = 1.0
REFINED_MARKER_LOCAL_CONTRAST_SHELL_OUTER_RADIUS = 4
REFINED_MARKER_BRIGHT_PIXEL_MEASURE = "fraction"
REFINED_MARKER_BRIGHT_PIXEL_THRESHOLD = 120
REFINED_MARKER_BRIGHT_PIXEL_MIN_COUNT = None
REFINED_MARKER_BRIGHT_PIXEL_MIN_FRACTION = 0.09

if REFINE_WITH_CACHED_CELLPOSE_OUTPUTS:
    refined_cell_model_config = replace(
        CELL_MODEL_CONFIG,
        postfilters=REFINED_CELL_POSTFILTERS,
        min_intensity_measure=REFINED_CELL_MIN_INTENSITY_MEASURE,
        min_intensity_threshold=REFINED_CELL_MIN_INTENSITY_THRESHOLD,
        local_contrast_k=REFINED_CELL_LOCAL_CONTRAST_K,
        local_contrast_shell_inner_radius=REFINED_CELL_LOCAL_CONTRAST_SHELL_INNER_RADIUS,
        local_contrast_shell_outer_radius=REFINED_CELL_LOCAL_CONTRAST_SHELL_OUTER_RADIUS,
        bright_pixel_measure=REFINED_CELL_BRIGHT_PIXEL_MEASURE,
        bright_pixel_threshold=REFINED_CELL_BRIGHT_PIXEL_THRESHOLD,
        bright_pixel_min_count=REFINED_CELL_BRIGHT_PIXEL_MIN_COUNT,
        bright_pixel_min_fraction=REFINED_CELL_BRIGHT_PIXEL_MIN_FRACTION,
    )
    refined_marker_model_config = replace(
        MARKER_MODEL_CONFIG,
        postfilters=REFINED_MARKER_POSTFILTERS,
        min_intensity_measure=REFINED_MARKER_MIN_INTENSITY_MEASURE,
        min_intensity_threshold=REFINED_MARKER_MIN_INTENSITY_THRESHOLD,
        local_contrast_k=REFINED_MARKER_LOCAL_CONTRAST_K,
        local_contrast_shell_inner_radius=REFINED_MARKER_LOCAL_CONTRAST_SHELL_INNER_RADIUS,
        local_contrast_shell_outer_radius=REFINED_MARKER_LOCAL_CONTRAST_SHELL_OUTER_RADIUS,
        bright_pixel_measure=REFINED_MARKER_BRIGHT_PIXEL_MEASURE,
        bright_pixel_threshold=REFINED_MARKER_BRIGHT_PIXEL_THRESHOLD,
        bright_pixel_min_count=REFINED_MARKER_BRIGHT_PIXEL_MIN_COUNT,
        bright_pixel_min_fraction=REFINED_MARKER_BRIGHT_PIXEL_MIN_FRACTION,
    )

    run_result = refine_run_result_from_cellpose_cache(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        colocalization_config=COLOCALIZATION_CONFIG,
        cell_model_config=refined_cell_model_config,
        marker_model_config=refined_marker_model_config,
        cell_cellprob_threshold=REFINED_CELL_CELLPROB_THRESHOLD,
        cell_flow_threshold=REFINED_CELL_FLOW_THRESHOLD,
        marker_cellprob_threshold=REFINED_MARKER_CELLPROB_THRESHOLD,
        marker_flow_threshold=REFINED_MARKER_FLOW_THRESHOLD,
        optional_region_result=None)

    print(run_result.tables.overview)

    if RUNTIME_CONFIG.open_results:
        result_viewer = show_analysis_results(
            loaded_images=loaded_images,
            roi_labels_2d=roi_labels_2d,
            run_result=run_result,
            display_names=DISPLAY_NAMES,
            optional_region_result=None,
            viewer=result_viewer,
            layers_to_show=REFINEMENT_RESULT_LAYER_KEYS,
            replace_existing_layers=True,
            show_optional_region_image=True,
        )

        print(f"Inspecting refined visualization for:\n{SELECTED_FILE_NAME}")
        napari.run()
else:
    print("Cached Cellpose refinement is disabled for this run.")
# %% EXPORT RESULTS
export_analysis_outputs(
    run_result=run_result,
    paths=loaded_images.paths,
    optional_region_result=None,
)
print("Final results exported.")

# %% END
