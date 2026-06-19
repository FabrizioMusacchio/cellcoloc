"""Interactive user script for whole-image microglia colocalization analysis.

This script is the whole-image counterpart of
``microglia_2D_user_script.py``. It uses the same channel definitions and core
pipeline, but skips interactive ROI drawing and analyzes the entire field of
view as one single ROI.

The core pipeline auto-detects whether each file is truly 2D or 3D by
inspecting the OME/TZCYX z dimension after loading with :mod:`omio`.
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
    export_analysis_outputs,
    load_analysis_images,
    run_roi_cellpose_colocalization,
    show_analysis_results,
)

import napari


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
    model_name_or_path="cyto3",  # cpsam for Cellpose 4, cyto3 for Cellpose 3
    segmentation_method="cellpose",
    do_3d=None,
    anisotropy=True,
    flow3d_smooth=0,
)

MARKER_MODEL_CONFIG = CellposeModelConfig(
    diameter=15,
    model_name_or_path="cyto3",  # cpsam for Cellpose 4, cyto3 for Cellpose 3
    segmentation_method="cellpose",
    do_3d=None,
    anisotropy=True,
    flow3d_smooth=0,
)

COLOCALIZATION_CONFIG = ColocalizationConfig(
    min_cell_voxels=50,
    overlap_fraction_threshold=0.02,
    min_overlap_voxels=10,
)

RUNTIME_CONFIG = RuntimeConfig(
    draw_rois=False,
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
DATA_PATH = DATA_DIR / SELECTED_FILE_NAME
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Selected file does not exist:\n{DATA_PATH}")


# %% LOAD THE ANALYSIS CHANNELS
loaded_images = load_analysis_images(
    source_path=DATA_PATH,
    channel_config=CHANNEL_CONFIG,
    voxel_scale_zyx=VOXEL_SCALE_ZYX,
    crop_for_testing=RUNTIME_CONFIG.crop_for_testing,
)

print(f"Results directory:\n{loaded_images.paths.results_dir}")


# %% CREATE A FULL-IMAGE ROI
roi_labels_2d = create_full_image_roi_labels(loaded_images.cell_image.shape[1:])
print("Whole-image mode is enabled. A single ROI covering the full image will be used.")


# %% RUN THE ROI-WISE CELLPOSE COLOCALIZATION AND EXPORT RESULTS
run_result = run_roi_cellpose_colocalization(
    loaded_images=loaded_images,
    roi_labels_2d=roi_labels_2d,
    cell_model_config=CELL_MODEL_CONFIG,
    marker_model_config=MARKER_MODEL_CONFIG,
    colocalization_config=COLOCALIZATION_CONFIG,
    runtime_config=RUNTIME_CONFIG,
    optional_region_result=None,
)

export_analysis_outputs(
    run_result=run_result,
    paths=loaded_images.paths,
    optional_region_result=None,
)

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
