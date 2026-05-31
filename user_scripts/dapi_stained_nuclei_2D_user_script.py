"""Interactive user script for ROI-based colocalization on the 2D nuclei demo.

This script targets the true 2D OME-TIFF dataset stored in
``example_data/dapi_stained_nuclei_2D``. The file currently contains two
channels and no reliable semantic channel names in the OME metadata, so the
channel assignment is made explicitly in the settings section below and can be
adjusted easily if needed.

The workflow is intentionally interactive and notebook-like:

1. choose and inspect the input file,
2. draw ROIs in napari,
3. save or reload those ROIs,
4. run Cellpose-based colocalization within the ROI set,
5. inspect the resulting masks and tables.
"""
# %% IMPORTS AND LOCAL PACKAGE BOOTSTRAP
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cell_coloc import (
    analyze_existing_masks,
    CellposeModelConfig,
    ChannelConfig,
    ColocalizationConfig,
    DisplayNames,
    RuntimeConfig,
    create_roi_drawing_viewer,
    extract_label_masks_from_viewer,
    export_analysis_outputs,
    load_analysis_images,
    load_roi_labels,
    refine_run_result_from_cellpose_cache,
    run_roi_cellpose_colocalization,
    save_roi_labels_from_shapes,
    show_analysis_results,
)

import napari
import numpy as np
# %% PROJECT SETTINGS
DATA_PATH = PROJECT_ROOT / "example_data" / "dapi_stained_nuclei_2D" / "dapi_stained_nuclei_2D.ome.tif"

CHANNEL_CONFIG = ChannelConfig(
    cell_channel=0,
    marker_channel=1,
    optional_region_channel=None,
)

DISPLAY_NAMES = DisplayNames(
    cell="Channel 0",
    marker="Channel 1",
    optional_region="Unused",
    positive_cells="Channel 0 + Channel 1 positive masks",
)

VOXEL_SCALE_ZYX = (1.0, 0.325, 0.325)

CELL_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",
    diameter=None,
    do_3d=None,
    cellprob_threshold=-0.5, # threshold for the cell probability map, between -6 and 6, where higher values lead to fewer cells (default: 0.0)
    flow_threshold=0.5,     # quality threshold; After mask generation, Cellpose checks whether the flows reconstructed from the mask are 
                            # consistent with the flows predicted by the network. The mean squared error between the two is used as the flow 
                            # error; masks with an error that is too large are discarded. The default value is 0.4.  
                            # The higher, the more tolerant the algorithm is, i.e. more cells but also more false positives.
)

MARKER_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",
    diameter=None,
    do_3d=None,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
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
# %% LOAD THE ANALYSIS CHANNELS
loaded_images = load_analysis_images(
    source_path=DATA_PATH,
    channel_config=CHANNEL_CONFIG,
    voxel_scale_zyx=VOXEL_SCALE_ZYX,
    crop_for_testing=RUNTIME_CONFIG.crop_for_testing)

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
    viewer = show_analysis_results(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        display_names=DISPLAY_NAMES,
        optional_region_result=None,
    )
    print("Inspect the final layers in napari and close the window when finished.")
    napari.run()
# %% OPTIONALLY REFINE RESULTS AND VISUALIZE UPDATED RESULT IN NAPARI
REFINE_WITH_CACHED_CELLPOSE_OUTPUTS = True
REFINED_CELL_CELLPROB_THRESHOLD     = CELL_MODEL_CONFIG.cellprob_threshold - 1.0
REFINED_CELL_FLOW_THRESHOLD         = CELL_MODEL_CONFIG.flow_threshold + 0.0
REFINED_MARKER_CELLPROB_THRESHOLD   = MARKER_MODEL_CONFIG.cellprob_threshold
REFINED_MARKER_FLOW_THRESHOLD       = MARKER_MODEL_CONFIG.flow_threshold

if REFINE_WITH_CACHED_CELLPOSE_OUTPUTS:
    run_result = refine_run_result_from_cellpose_cache(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        colocalization_config=COLOCALIZATION_CONFIG,
        cell_cellprob_threshold=REFINED_CELL_CELLPROB_THRESHOLD,
        cell_flow_threshold=REFINED_CELL_FLOW_THRESHOLD,
        marker_cellprob_threshold=REFINED_MARKER_CELLPROB_THRESHOLD,
        marker_flow_threshold=REFINED_MARKER_FLOW_THRESHOLD,
        optional_region_result=None)

    print(run_result.tables.overview)

    viewer = show_analysis_results(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        display_names=DISPLAY_NAMES,
        optional_region_result=None)
    print("Inspect the refined layers in napari and close the window when finished.")
    napari.run()
else:
    print("Cached Cellpose refinement is disabled for this run.")
# %% OPTIONALLY REANALYZE MANUALLY EDITED LABEL LAYERS FROM NAPARI
REANALYZE_EDITED_LABELS_FROM_VIEWER = True

if REANALYZE_EDITED_LABELS_FROM_VIEWER:
    cell_masks_from_viewer, marker_masks_from_viewer = extract_label_masks_from_viewer(viewer)
    run_result = analyze_existing_masks(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        cell_masks=cell_masks_from_viewer,
        marker_masks=marker_masks_from_viewer,
        colocalization_config=COLOCALIZATION_CONFIG,
        optional_region_result=None,
        cell_refinement_context=run_result.cell_refinement_context,
        marker_refinement_context=run_result.marker_refinement_context,
    )

    print(run_result.tables.overview)

    viewer = show_analysis_results(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        display_names=DISPLAY_NAMES,
        optional_region_result=None,
    )
    print("Inspect the relabeled result in napari and close the window when finished.")
    napari.run()
else:
    print("Manual label reanalysis from the napari viewer is disabled for this run.")
# %% EXPORT RESULTS
export_analysis_outputs(
    run_result=run_result,
    paths=loaded_images.paths,
    optional_region_result=None)
print("Final results exported.")
# %% END
