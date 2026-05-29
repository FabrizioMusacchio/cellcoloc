"""Interactive user script for whole-image microglia colocalization analysis.

This script is configured for the datasets stored in
``example_data/microglia_2D``. Despite the folder name, the core pipeline now
auto-detects whether each file is truly 2D or 3D by inspecting the OME/TZCYX
z dimension after loading with :mod:`omio`.

The biological colocalization task in this script is:

- channel 0: ``Cx3cr1-tdTomato`` microglia reporter signal,
- channel 1: ``Iba1`` staining,
- channel 2: ``DAPI`` only for orientation and optional visualization.

The quantitative colocalization is performed between channel 0 and channel 1.
The full field of view is processed as one ROI, so no manual ROI drawing and
no optional third-channel thresholding are used here.
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
import pandas as pd
# %% PROJECT SETTINGS
DATA_DIR = PROJECT_ROOT / "example_data" / "microglia_2D"
DATA_PATHS = sorted(DATA_DIR.glob("*.czi"))
VISUALIZE_FILE_NAME = DATA_PATHS[0].name if DATA_PATHS else None

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
    model_name_or_path="cyto3", #cpsam for cellpose 4, cyto3 for cellpose 3
    do_3d=None,
)

MARKER_MODEL_CONFIG = CellposeModelConfig(
    diameter=15,
    model_name_or_path="cyto3", #cpsam for cellpose 4, cyto3 for cellpose 3
    do_3d=None,
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
# %% RUN THE WHOLE-IMAGE ANALYSIS FOR ALL FILES
overview_tables: list[pd.DataFrame] = []
summary_tables: list[pd.DataFrame] = []
detailed_tables: list[pd.DataFrame] = []

visualization_loaded_images = None
visualization_roi_labels_2d = None
visualization_run_result = None

for data_path in DATA_PATHS:
    # data_path = DATA_PATHS[0]  # Uncomment for quick testing of a single file
    print(f"\n{'=' * 80}\nProcessing file:\n{data_path}\n{'=' * 80}")

    loaded_images = load_analysis_images(
        source_path=data_path,
        channel_config=CHANNEL_CONFIG,
        voxel_scale_zyx=VOXEL_SCALE_ZYX,
        crop_for_testing=RUNTIME_CONFIG.crop_for_testing,
    )

    roi_labels_2d = create_full_image_roi_labels(loaded_images.cell_image.shape[1:])
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

    overview_table = run_result.tables.overview.copy()
    summary_table = run_result.tables.summary.copy()
    detailed_table = run_result.tables.detailed.copy()

    overview_table.insert(0, "source_file", data_path.name)
    summary_table.insert(0, "source_file", data_path.name)
    detailed_table.insert(0, "source_file", data_path.name)

    overview_tables.append(overview_table)
    summary_tables.append(summary_table)
    detailed_tables.append(detailed_table)

    if VISUALIZE_FILE_NAME == data_path.name:
        visualization_loaded_images = loaded_images
        visualization_roi_labels_2d = roi_labels_2d
        visualization_run_result = run_result
# %% COMBINE PROJECT-LEVEL TABLES
project_overview = pd.concat(overview_tables, ignore_index=True) if overview_tables else pd.DataFrame()
project_summary = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
project_detailed = pd.concat(detailed_tables, ignore_index=True) if detailed_tables else pd.DataFrame()

print("\nProject overview:")
print(project_overview)

project_results_dir = DATA_DIR / "results"
project_results_dir.mkdir(parents=True, exist_ok=True)
project_overview.to_csv(project_results_dir / "microglia_project_overview.csv", index=False)
project_summary.to_csv(project_results_dir / "microglia_project_cell_summary.csv", index=False)
project_detailed.to_csv(project_results_dir / "microglia_project_detailed_overlaps.csv", index=False)


# %% OPTIONAL TABLE INSPECTION
project_summary.head()


# %% VISUALIZE ONE SELECTED FILE IN NAPARI
if RUNTIME_CONFIG.open_results:
    if visualization_loaded_images is None or visualization_run_result is None or visualization_roi_labels_2d is None:
        raise ValueError(
            "No visualization payload is available. Check VISUALIZE_FILE_NAME "
            "or ensure at least one file was processed successfully."
        )

    viewer = show_analysis_results(
        loaded_images=visualization_loaded_images,
        roi_labels_2d=visualization_roi_labels_2d,
        run_result=visualization_run_result,
        display_names=DISPLAY_NAMES,
        optional_region_result=None,
    )

    if visualization_loaded_images.optional_region_image is not None:
        viewer.add_image(
            visualization_loaded_images.optional_region_image,
            name="DAPI",
            scale=visualization_loaded_images.voxel_scale_zyx,
            blending="additive",
            colormap="yellow",
            channel_axis=None,
        )

    print(f"Inspecting visualization for:\n{VISUALIZE_FILE_NAME}")
    napari.run()

# %% END
