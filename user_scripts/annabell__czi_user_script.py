"""Interactive example user script for the reusable ``cell_coloc`` pipeline.

This script mirrors the original prototype workflow, but all reusable core
logic now lives inside the :mod:`cell_coloc` package. The intended usage is to
run the file cell by cell in the VS Code interactive window:

1. adjust dataset paths and analysis settings in the configuration cell,
2. load the microscopy channels,
3. draw ROIs in napari and save them,
4. optionally segment a third channel for region coverage,
5. run Cellpose-based colocalization inside the ROIs,
6. inspect tables and masks in napari.

The default package design supports a generic two-channel problem:

- one channel for the larger cell object to segment,
- one channel for the marker object that defines positivity.

An optional third channel can be enabled to quantify thresholded region
coverage, as done here for tumor infiltration. All generated files are written
to a ``results`` subfolder located next to the raw dataset.
"""

# %% IMPORTS AND LOCAL PACKAGE BOOTSTRAP
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CACHE_ROOT = Path(tempfile.gettempdir()) / "cell_coloc_runtime_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "numba").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg_cache").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "napari").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_ROOT / "numba"))
os.environ.setdefault("NAPARI_CONFIG", str(CACHE_ROOT / "napari" / "settings.yaml"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg_cache"))

import napari

from cell_coloc import (
    CellposeModelConfig,
    ChannelConfig,
    ColocalizationConfig,
    DisplayNames,
    OptionalRegionSegmentationConfig,
    RuntimeConfig,
    create_roi_drawing_viewer,
    export_analysis_outputs,
    load_analysis_images,
    load_roi_labels,
    run_roi_cellpose_colocalization,
    save_roi_labels_from_shapes,
    segment_optional_region,
    show_analysis_results,
)
# %% PROJECT SETTINGS
DATA_PATH = PROJECT_ROOT / "example_data" / "czi_private_3D" / "ID24137_3rdsection_contralateralCtx_DAPI-GB-NeuN_20x.czi"

CHANNEL_CONFIG = ChannelConfig(
    cell_channel=2,
    marker_channel=1,
    optional_region_channel=0,
)

DISPLAY_NAMES = DisplayNames(
    cell="Neurons",
    marker="DAPI / nuclei",
    optional_region="Cancer infiltration",
    positive_cells="DAPI-positive neuron masks",
)

VOXEL_SCALE_ZYX = (3.0, 0.3899, 0.3899)

CELL_MODEL_CONFIG = CellposeModelConfig(
    diameter=60,
    model_name_or_path="cpsam", # cyto3 for cellpose 3, cpsam for cellpose 4
    # Example for a custom trained Cellpose model:
    # model_name_or_path="/absolute/path/to/custom_model",
)

MARKER_MODEL_CONFIG = CellposeModelConfig(
    diameter=20,
    model_name_or_path="cpsam", # nuclei for cellpose 3, cpsam for cellpose 4
)

COLOCALIZATION_CONFIG = ColocalizationConfig(
    min_cell_voxels=200,
    overlap_fraction_threshold=0.02,
    min_overlap_voxels=20,
)

OPTIONAL_REGION_CONFIG = OptionalRegionSegmentationConfig(
    enabled=True,
    method="li",
    percentile=98.0,
    gaussian_sigma=1.0,
    background_sigma=None,
    min_object_voxels=10,
    min_hole_voxels=10,
    apply_closing=True,
)

RUNTIME_CONFIG = RuntimeConfig(
    draw_rois=True,
    process_rois=True,
    open_results=True,
    use_gpu=True,
    # Use a very small crop for quick tests, for example:
    # crop_for_testing=(slice(0, 16), slice(0, 20), slice(0, 20)),
    crop_for_testing=None,
)
# %% LOAD THE ANALYSIS CHANNELS
loaded_images = load_analysis_images(
    source_path=DATA_PATH,
    channel_config=CHANNEL_CONFIG,
    voxel_scale_zyx=VOXEL_SCALE_ZYX,
    crop_for_testing=RUNTIME_CONFIG.crop_for_testing,
)

print(f"Results directory:\n{loaded_images.paths.results_dir}")


# %% DRAW ROIS INTERACTIVELY IN NAPARI
if RUNTIME_CONFIG.draw_rois:
    viewer, shapes_layer = create_roi_drawing_viewer(
        loaded_images=loaded_images,
        display_names=DISPLAY_NAMES,
    )
    print("Draw ROIs in napari and close the window. Then run the next cell.")
    napari.run()
else:
    print("ROI drawing is disabled. The next cell will load an existing ROI mask from disk.")


# %% SAVE THE DRAWN ROIS OR LOAD AN EXISTING ROI MASK
if RUNTIME_CONFIG.draw_rois:
    roi_labels_2d = save_roi_labels_from_shapes(
        shapes_layer=shapes_layer,
        output_path=loaded_images.paths.roi_mask_path,
        image_shape_yx=loaded_images.cell_image.max(axis=0).shape,
        scale_yx=loaded_images.voxel_scale_zyx[1:],
    )
else:
    roi_labels_2d = load_roi_labels(loaded_images.paths.roi_mask_path)

roi_ids = np.unique(roi_labels_2d)
roi_ids = roi_ids[roi_ids != 0]
print(f"ROI ids: {roi_ids}")


# %% OPTIONAL THIRD-CHANNEL SEGMENTATION
optional_region_result = None
if OPTIONAL_REGION_CONFIG.enabled:
    if loaded_images.optional_region_image is None:
        raise ValueError(
            "The optional third-channel analysis is enabled, but no optional "
            "region channel was configured."
        )

    optional_region_result = segment_optional_region(
        image_zyx=loaded_images.optional_region_image,
        roi_labels_2d=roi_labels_2d,
        config=OPTIONAL_REGION_CONFIG,
    )
    print(f"Optional region threshold: {optional_region_result.threshold}")
else:
    print("Optional third-channel analysis is disabled. Continuing with the two-channel workflow only.")


# %% RUN THE ROI-WISE CELLPOSE COLOCALIZATION AND EXPORT RESULTS
if RUNTIME_CONFIG.process_rois:
    run_result = run_roi_cellpose_colocalization(
        loaded_images       =loaded_images,
        roi_labels_2d       =roi_labels_2d,
        cell_model_config   =CELL_MODEL_CONFIG,
        marker_model_config =MARKER_MODEL_CONFIG,
        colocalization_config=COLOCALIZATION_CONFIG,
        runtime_config      =RUNTIME_CONFIG,
        optional_region_result=optional_region_result,
    )

    export_analysis_outputs(
        run_result=run_result,
        paths=loaded_images.paths,
        optional_region_result=optional_region_result,
    )

    print(run_result.tables.overview)
else:
    print("ROI processing is disabled in the runtime settings.")
# %% OPTIONAL TABLE INSPECTION IN THE INTERACTIVE WINDOW
if RUNTIME_CONFIG.process_rois:
    run_result.tables.summary.head()
# %% OPEN THE FINAL ANALYSIS RESULT IN NAPARI
if RUNTIME_CONFIG.open_results and RUNTIME_CONFIG.process_rois:
    viewer = show_analysis_results(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        display_names=DISPLAY_NAMES,
        optional_region_result=optional_region_result,
    )
    print("Inspect the final layers in napari and close the window when finished.")
    napari.run()
# %% END