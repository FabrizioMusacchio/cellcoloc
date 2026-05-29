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
    run_roi_cellpose_colocalization,
    save_roi_labels_from_shapes,
    show_analysis_results,
)

import napari
import numpy as np
# %% PROJECT SETTINGS
DATA_DIR = PROJECT_ROOT / "example_data" / "microglia_2D"
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
    positive_cells="tdTomato + Iba1 positive masks",
)

# Placeholder values. Replace with the true metadata-derived pixel sizes when
# available for quantitative micrometer-scale interpretation.
VOXEL_SCALE_ZYX = (1.0, 1.0, 1.0)

CELL_MODEL_CONFIG = CellposeModelConfig(
    diameter=15,
    model_name_or_path="cpsam",  # cpsam for Cellpose 4, cyto3 for Cellpose 3
    do_3d=None,
)

MARKER_MODEL_CONFIG = CellposeModelConfig(
    diameter=15,
    model_name_or_path="cpsam",  # cpsam for Cellpose 4, cyto3 for Cellpose 3
    do_3d=None,
)

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
# %% DRAW ROIS INTERACTIVELY IN NAPARI
if USE_FULL_IMAGE_AS_SINGLE_ROI:
    print("Whole-image mode is enabled. ROI drawing is skipped.")
elif RUNTIME_CONFIG.draw_rois:
    viewer, shapes_layer = create_roi_drawing_viewer(
        loaded_images=loaded_images,
        display_names=DISPLAY_NAMES,
    )
    print("Draw ROIs in napari and close the window. Then run the next cell.")
    napari.run()
else:
    print("ROI drawing is disabled. The next cell will load an existing ROI mask from disk.")
# %% SAVE THE DRAWN ROIS OR LOAD AN EXISTING ROI MASK
if USE_FULL_IMAGE_AS_SINGLE_ROI:
    roi_labels_2d = create_full_image_roi_labels(loaded_images.cell_image.shape[1:])
elif RUNTIME_CONFIG.draw_rois:
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
# %% RUN THE ROI-WISE CELLPOSE COLOCALIZATION AND EXPORT RESULTS
run_result = run_roi_cellpose_colocalization(
    loaded_images=loaded_images,
    roi_labels_2d=roi_labels_2d,
    cell_model_config=CELL_MODEL_CONFIG,
    marker_model_config=MARKER_MODEL_CONFIG,
    colocalization_config=COLOCALIZATION_CONFIG,
    runtime_config=RUNTIME_CONFIG,
    optional_region_result=None)

export_analysis_outputs(
    run_result=run_result,
    paths=loaded_images.paths,
    optional_region_result=None)

print(run_result.tables.overview)
# %% OPTIONAL TABLE INSPECTION
run_result.tables.summary.head()


# %% VISUALIZE THE RESULT IN NAPARI
if RUNTIME_CONFIG.open_results:
    viewer = show_analysis_results(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        display_names=DISPLAY_NAMES,
        optional_region_result=None,
    )

    if loaded_images.optional_region_image is not None:
        viewer.add_image(
            loaded_images.optional_region_image,
            name="DAPI",
            scale=loaded_images.voxel_scale_zyx,
            blending="additive",
            colormap="yellow",
            channel_axis=None,
        )

    print(f"Inspecting visualization for:\n{SELECTED_FILE_NAME}")
    napari.run()

# %% END
