"""Interactive user script for ROI-based colocalization on the 2D nuclei demo.

This script targets the true 2D OME-TIFF dataset stored in
``example_data/dapi_stained_nuclei_2D``. The file currently contains two
channels and no reliable semantic channel names in the OME metadata, so the
channel assignment is made explicitly in the settings section below and can be
adjusted easily if needed.

The workflow is intentionally interactive and notebook-like:

1. choose and inspect the input file,
2. optionally analyze the whole image as one ROI or draw custom ROIs in napari,
3. optionally reuse a previously saved ROI mask from the results directory,
4. run Cellpose-based colocalization within the ROI set,
5. inspect the resulting masks and tables.

author: Fabrizio Musacchio
date:   June 2026
"""
# %% IMPORTS
# get the project root for accessing the example dataset which is 
# located in the parent folder of the user_scripts folder:
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# import relevant CellColoc functions for this script:
from cellcoloc import (
    analyze_existing_masks,
    CellposeModelConfig,
    ChannelConfig,
    ColocalizationConfig,
    DisplayNames,
    RuntimeConfig,
    create_full_image_roi_labels,
    create_roi_drawing_viewer,
    extract_label_masks_from_viewer,
    export_analysis_outputs,
    load_analysis_images,
    load_roi_labels,
    refine_run_result_from_cellpose_cache,
    run_roi_cellpose_colocalization,
    save_roi_labels_from_shapes,
    show_analysis_results,
    try_load_roi_labels)

# additional packages:
import napari
import numpy as np
# %% PROJECT SETTINGS

# define path to the dataset file relative to the project root:
DATA_PATH = PROJECT_ROOT / "example_data" / "dapi_stained_nuclei_2D" / "dapi_stained_nuclei_2D.ome.tif"

# define the channel configuration for this dataset:
CHANNEL_CONFIG = ChannelConfig(
    cell_channel=0,
    marker_channel=1,
    optional_region_channel=None)

# define display names for the channels and result layers:
DISPLAY_NAMES = DisplayNames(
    cell="Channel 0",
    marker="Channel 1",
    optional_region="Unused",
    positive_cells="Channel 0 + Channel 1 positive masks")

# optional: define the voxel scale in ZYX (for 3D) or YX (for 2D) order. 
# Set this to None to use the voxel scale from the OME metadata, if available.
#VOXEL_SCALE_ZYX = (1.0, 0.325, 0.325) # CellColoc allows both ZYX and YX even for 2D data. 
VOXEL_SCALE_ZYX = (0.325, 0.325)

# define the Cellpose model configuration for the cell and marker segmentation:
CELL_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",     # 'cpsam' for cellpose 4, 'cty3', 'cyto2' or 'nuclei' for cellpose 3
    segmentation_method="cellpose", # 'cellpose' or: 'otsu', 'li', or 'percentile' for marker segmentation
    diameter=None,                  # if None, Cellpose will estimate the diameter from the data. Adjust if you know the expected cell/nuclei size in pixels.
    do_3d=None,                     # whether Cellpose should run in 3D mode; set to None to let CellColoc decide based on the input data shape
    cellprob_threshold=-0.5, # threshold for the cell probability map, between -6 and 6, where higher values lead to fewer cells (default: 0.0)
    flow_threshold=0.5,     # quality threshold; After mask generation, Cellpose checks whether the flows reconstructed from the mask are 
                            # consistent with the flows predicted by the network. The mean squared error between the two is used as the flow 
                            # error; masks with an error that is too large are discarded. The default value is 0.4.  
                            # The higher, the more tolerant the algorithm is, i.e. more cells but also more false positives.
)

# define the Cellpose model configuration for the marker segmentation. You can 
# use the same or a different model as for the cell segmentation, depending 
# on your data and preferences.
MARKER_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",
    segmentation_method="cellpose",
    diameter=None,
    do_3d=None,
    cellprob_threshold=0.0,
    flow_threshold=0.4)

# define the colocalization configuration:
COLOCALIZATION_CONFIG = ColocalizationConfig(
    min_cell_voxels=50,                 # minimum number of voxels for a cell to be included in the analysis; adjust based on your expected cell/nuclei size and the image resolution
    overlap_fraction_threshold=0.02,    # minimum fraction of overlap between two cells to be considered colocalized
    min_overlap_voxels=10)              # minimum number of overlapping voxels to be considered colocalized

# define the runtime configuration:
RUNTIME_CONFIG = RuntimeConfig(
    draw_rois=True,         # whether to draw ROIs in napari; if False, the whole image will be analyzed as one ROI
    process_rois=True,      # whether to run the analysis for each ROI; if False, drawn ROIs will be ignored and the whole image will be analyzed as one ROI
    open_results=True,      # whether to open the results after the analysis is complete
    use_gpu=True,           # whether to use GPU for the analysis
    crop_for_testing=None)  # whether to crop the image for testing purposes; if None, no cropping is applied; 
                            # set as a tuple of slices, e.g. (slice(0, 16), slice(0, 20), slice(0, 20)) for a 
                            # small 3D crop; for 2D, set the first slice to slice(0, 1), e.g. 
                            # (slice(0, 1), slice(0, 20), slice(0, 20))

# additional ROI settings: 
USE_FULL_IMAGE_AS_SINGLE_ROI = True         # if True, the whole image will be analyzed as one ROI and the 
                                            # ROI drawing step will be skipped.
REUSE_EXISTING_ROI_MASK_IF_AVAILABLE = True # if True, the script will check for an existing ROI mask in 
                                            #the results directory and reuse it if found, skipping the ROI 
                                            # drawing step.
# for internal house-keeping, we need to define a variable for storing the existing ROI labels if we want 
# to reuse them:
existing_roi_labels = None
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
elif REUSE_EXISTING_ROI_MASK_IF_AVAILABLE:
    existing_roi_labels = try_load_roi_labels(loaded_images.paths.roi_mask_path)
    if existing_roi_labels is not None:
        print("Found an existing ROI mask in the results directory. "
              "ROI drawing is skipped and the saved mask will be reused.")
    elif RUNTIME_CONFIG.draw_rois:
        viewer, shapes_layer = create_roi_drawing_viewer(
            loaded_images=loaded_images,
            display_names=DISPLAY_NAMES)
        print("Draw ROIs in napari and close the window. Then run the next cell.")
        napari.run()
    else:
        print("No saved ROI mask was found and ROI drawing is disabled. "
              "The next cell will fail unless you enable drawing or whole-image mode.")
elif RUNTIME_CONFIG.draw_rois:
    viewer, shapes_layer = create_roi_drawing_viewer(
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
# %% RUN THE ROI-WISE CELLPOSE COLOCALIZATION
run_result = run_roi_cellpose_colocalization(
    loaded_images           = loaded_images,
    roi_labels_2d           = roi_labels_2d,
    cell_model_config       = CELL_MODEL_CONFIG,
    marker_model_config     = MARKER_MODEL_CONFIG,
    colocalization_config   = COLOCALIZATION_CONFIG,
    runtime_config          = RUNTIME_CONFIG,
    optional_region_result  = None)
# print the first 3 columns of the overview table for a quick check:
print("Overview of the colocalization results (first 3 columns):")
print(run_result.tables.overview.iloc[:, :3])
# %% VISUALIZE THE RESULT IN NAPARI
if RUNTIME_CONFIG.open_results:
    viewer = show_analysis_results(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        display_names=DISPLAY_NAMES,
        optional_region_result=None)
    print("Inspect the final layers in napari and close the window when finished.")
    napari.run()
# %% OPTIONALLY REFINE RESULTS AND VISUALIZE UPDATED RESULT IN NAPARI
REFINE_WITH_CACHED_CELLPOSE_OUTPUTS = True
REFINED_CELL_CELLPROB_THRESHOLD     = CELL_MODEL_CONFIG.cellprob_threshold - 5.0
REFINED_CELL_FLOW_THRESHOLD         = CELL_MODEL_CONFIG.flow_threshold - 0.3
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

    print(run_result.tables.overview.iloc[:, :3])

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

""" 
In the opened Napari viewer, you can manually edit the Cellpose-generated label layers 
(e.g. using the brush, eraser, or other label editing tools) to correct any segmentation 
errors. When you are done, execute this cell to extract the edited masks from the viewer. 
The colocalization analysis will be re-run based on the edited masks, and the updated results 
will be displayed in a new Napari viewer instance.
"""

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

    print(run_result.tables.overview.iloc[:, :3])

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
