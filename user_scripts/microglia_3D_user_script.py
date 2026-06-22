"""Interactive user script for microglia colocalization analysis.

This script is configured for the datasets stored in
``example_data/microglia_3D``. 

The biological colocalization task in this script is:

- channel 0: ``Cx3cr1-tdTomato`` microglia reporter signal,
- channel 1: ``Iba1`` staining,
- channel 2: ``DAPI`` only for orientation and optional visualization.

The quantitative colocalization is performed between channel 0 and channel 1.
ROIs can be drawn interactively in napari or, if desired, the whole field of
view can be analyzed as one single ROI.

author: Fabrizio Musacchio
date:   June 2026
"""
# %% IMPORTS AND LOCAL PACKAGE BOOTSTRAP
# get the project root for accessing the example dataset which is 
# located in the parent folder of the user_scripts folder:
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# import relevant CellColoc functions for this script:
from cellcoloc import (
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
    analyze_existing_masks,
    extract_label_masks_from_viewer)

# additional packages:
import napari
import numpy as np
from dataclasses import replace
# %% PROJECT SETTINGS

# define path to the dataset file relative to the project root:
DATA_DIR = PROJECT_ROOT / "example_data" / "microglia_3D"
# "glob" all image files in the directory:
DATA_PATHS = sorted(DATA_DIR.glob("*"))
allowed_extensions = [".czi", ".tif", ".tiff", ".ome.tif", ".ome.tiff"]
DATA_PATHS = [p for p in DATA_PATHS if p.suffix.lower() in allowed_extensions]
# remove .-prefixed hidden files that some operating systems create:
DATA_PATHS = [p for p in DATA_PATHS if not p.name.startswith(".")]

# define the channel configuration for this dataset:
CHANNEL_CONFIG = ChannelConfig(
    cell_channel=0,
    marker_channel=1,
    optional_region_channel=2)

# define display names for the channels and result layers:
DISPLAY_NAMES = DisplayNames(
    cell="Cx3cr1-tdTomato",
    marker="Iba1",
    optional_region="DAPI",
    positive_cells="tdTomato + Iba1 positive masks")

# optional: define the voxel scale in ZYX (for 3D) or YX (for 2D) order. 
# Set this to None to use the voxel scale from the OME metadata, if available.
#VOXEL_SCALE_ZYX = (1.0, 0.6239258, 0.6239258) # CellColoc allows both ZYX and YX even for 2D data. 
VOXEL_SCALE_ZYX = None#(1.0, 0.6239258, 0.6239258)

# define the Cellpose model configuration for the cell and marker segmentation:
CELL_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",     # 'cpsam' for cellpose 4, 'cty3', 'cyto2' or 'nuclei' for cellpose 3
    segmentation_method="cellpose", # 'cellpose' or: 'otsu', 'li', or 'percentile' for marker segmentation
    diameter=None,                  # if None, Cellpose will estimate the diameter from the data. Adjust if you know the expected cell/nuclei size in pixels.
    z_crop=None,                    # optional global analysis z-crop as (start, stop); applies to all channels and all ROIs
    anisotropy=True,                # Cellpose anisotropy correction based on the voxel scale z/y ratio; set to False/True to disable/enable 
    flow3d_smooth=3,                # Gaussian smoothing for 3D flow fields; int, default: 0, range: 0-10
    prefilter="gaussian",           # available options: "gaussian", "laplacian_of_gaussian"/"log", "median", None
        prefilter_sigma_xy=0.8,     # Gaussian prefilter sigma in xy; float, default: 0.0 (no prefilter), range: 0.0-10.0
        prefilter_sigma_z=0.0,      # Gaussian prefilter sigma in z; float, default: 0.0 (no prefilter), range: 0.0-10.0
        prefilter_median_size_xy=3, # Median prefilter size in xy; int, default: 0 (no prefilter), range: 0-10
        prefilter_median_size_z=3,  # Median prefilter size in z; int, default: 0 (no prefilter), range: 0-10
    postfilters=None,                   # available options: "min_intensity", "local_contrast", "bright_pixel_support", None
        min_intensity_measure="mean",   # measure for min intensity postfilter; available options: "mean", "max", "median"
        min_intensity_threshold=None,   # threshold for min intensity postfilter; float, no default, range depends on the image data
        local_contrast_k=1.0,           # k for local contrast postfilter; float, default: 1.0, range: 0.0-10.0
        local_contrast_shell_inner_radius=1, # local contrast shell inner radius in pixels; int, default: 1, range: 0-10
        local_contrast_shell_outer_radius=4, # local contrast shell outer radius in pixels; int, default: 4, range: 0-20
        bright_pixel_measure="count",   # measure for bright pixel support postfilter; available options: "count", "fraction"
        bright_pixel_threshold=None,    # threshold for bright pixel support postfilter; float, no default, range depends on the image data
        bright_pixel_min_count=None,    # minimum count for bright pixel support postfilter; int, no default, range depends on the image data
        bright_pixel_min_fraction=None, # minimum fraction for bright pixel support postfilter; float, no default, range depends on the image data
    cellprob_threshold=1.5,         # threshold for the cell probability map, between -6 and 6, where higher values lead to fewer cells (default: 0.0)
    flow_threshold=0.4,             # quality threshold; After mask generation, Cellpose checks whether the flows reconstructed from the mask are 
                                    # consistent with the flows predicted by the network. The mean squared error between the two is used as the flow 
                                    # error; masks with an error that is too large are discarded. The default value is 0.4.  
                                    # The higher, the more tolerant the algorithm is, i.e. more cells but also more false positives.
    )

# define the Cellpose model configuration for the cell and marker segmentation:
MARKER_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam", 
    segmentation_method="otsu", 
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

# define the colocalization configuration:
COLOCALIZATION_CONFIG = ColocalizationConfig(
    min_cell_voxels=50,
    overlap_fraction_threshold=0.02,
    min_overlap_voxels=10)

# define the runtime configuration:
RUNTIME_CONFIG = RuntimeConfig(
    draw_rois=True,
    process_rois=True,
    open_results=True,
    use_gpu=True,
    crop_for_testing=None,
    image_loading_mode="memap",  # available options: "memory", "memap"
)

USE_FULL_IMAGE_AS_SINGLE_ROI = False
REUSE_EXISTING_ROI_MASK_IF_AVAILABLE = True
INITIAL_RESULT_LAYER_KEYS = [ # controls which layers are shown in napari after the initial analysis run;
    "cell_image",
    "marker_image",
    "optional_region_image",
    "rois",
    "roi_numbers",
    "cell_masks",
    "marker_masks",
    "positive_cells"]
REFINEMENT_RESULT_LAYER_KEYS = [ # controls which layers are shown in napari after the optional refinement based on cached Cellpose outputs;
    "cell_masks",
    "marker_masks",
    "positive_cells",
    "rois"]

# select the dataset file to analyze if there are multiple files in the input directory:
print("Detected input files:")
for data_path in DATA_PATHS:
    print(f" - {data_path.name}")
SELECTED_FILE_NAME = DATA_PATHS[0].name
DATA_PATH = DATA_DIR / SELECTED_FILE_NAME
print(f"Selected file for analysis:\n{DATA_PATH}")
# %% LOAD THE ANALYSIS CHANNELS
loaded_images = load_analysis_images(
    source_path=DATA_PATH,
    channel_config=CHANNEL_CONFIG,
    voxel_scale_zyx=VOXEL_SCALE_ZYX,
    crop_for_testing=RUNTIME_CONFIG.crop_for_testing,
    image_loading_mode=RUNTIME_CONFIG.image_loading_mode)
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
# %% RUN THE ROI-WISE SEGMENTATION AND COLOCALIZATION ANALYSIS
run_result = run_roi_cellpose_colocalization(
    loaded_images=loaded_images,
    roi_labels_2d=roi_labels_2d,
    cell_model_config=CELL_MODEL_CONFIG,
    marker_model_config=MARKER_MODEL_CONFIG,
    colocalization_config=COLOCALIZATION_CONFIG,
    runtime_config=RUNTIME_CONFIG,
    optional_region_result=None)
print("Initial analysis finished. "
      f"Overview rows: {len(run_result.tables.overview)}, "
      f"summary rows: {len(run_result.tables.summary)}.")
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
# %% OPTIONALLY SET OR UPDATE A GLOBAL Z CROP FOR SUBSEQUENT REFINEMENT
REFINEMENT_ANALYSIS_Z_CROP = (5,20)
#REFINEMENT_ANALYSIS_Z_CROP = None
# Example: (5, 24) or None
# `None` keeps the current analysis z range unchanged.
# A tuple such as `(z_start, z_stop)` restricts all subsequent internal
# refinement calculations to that z interval for all channels and all ROIs.
if REFINEMENT_ANALYSIS_Z_CROP is None:
    print(f"No refinement z-crop requested. Subsequent refinement will keep the "
          f"current analysis z range: {run_result.analysis_z_bounds}.")
else:
    print(f"Subsequent refinement will use this global analysis z-crop:\n"
          f"{REFINEMENT_ANALYSIS_Z_CROP}")
# %% OPTIONALLY REFINE RESULTS AND VISUALIZE UPDATED RESULT IN NAPARI
REFINE_WITH_CACHED_CELLPOSE_OUTPUTS = True

REFINED_CELL_CELLPROB_THRESHOLD = CELL_MODEL_CONFIG.cellprob_threshold- 3.0 # - 0.9
REFINED_CELL_FLOW_THRESHOLD = CELL_MODEL_CONFIG.flow_threshold-0.2 # 0.1

REFINED_MARKER_CELLPROB_THRESHOLD = -3#4.5
REFINED_MARKER_FLOW_THRESHOLD = 0.4#0.8

REFINED_CELL_POSTFILTERS = ["min_intensity", "bright_pixel_support", "local_contrast"] # available options: "min_intensity", "local_contrast", "bright_pixel_support", None
REFINED_CELL_MIN_INTENSITY_MEASURE = "max"
REFINED_CELL_MIN_INTENSITY_THRESHOLD = 250
REFINED_CELL_LOCAL_CONTRAST_K = 4
REFINED_CELL_LOCAL_CONTRAST_SHELL_INNER_RADIUS = 1.0
REFINED_CELL_LOCAL_CONTRAST_SHELL_OUTER_RADIUS = 10
REFINED_CELL_BRIGHT_PIXEL_MEASURE = "fraction"
REFINED_CELL_BRIGHT_PIXEL_THRESHOLD = 110
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

effective_refinement_z_crop = (
    CELL_MODEL_CONFIG.z_crop if REFINEMENT_ANALYSIS_Z_CROP is None else REFINEMENT_ANALYSIS_Z_CROP)

if REFINE_WITH_CACHED_CELLPOSE_OUTPUTS:
    refined_cell_model_config = replace(
        CELL_MODEL_CONFIG,
        z_crop=effective_refinement_z_crop,
        postfilters=REFINED_CELL_POSTFILTERS,
        min_intensity_measure=REFINED_CELL_MIN_INTENSITY_MEASURE,
        min_intensity_threshold=REFINED_CELL_MIN_INTENSITY_THRESHOLD,
        local_contrast_k=REFINED_CELL_LOCAL_CONTRAST_K,
        local_contrast_shell_inner_radius=REFINED_CELL_LOCAL_CONTRAST_SHELL_INNER_RADIUS,
        local_contrast_shell_outer_radius=REFINED_CELL_LOCAL_CONTRAST_SHELL_OUTER_RADIUS,
        bright_pixel_measure=REFINED_CELL_BRIGHT_PIXEL_MEASURE,
        bright_pixel_threshold=REFINED_CELL_BRIGHT_PIXEL_THRESHOLD,
        bright_pixel_min_count=REFINED_CELL_BRIGHT_PIXEL_MIN_COUNT,
        bright_pixel_min_fraction=REFINED_CELL_BRIGHT_PIXEL_MIN_FRACTION)
    refined_marker_model_config = replace(
        MARKER_MODEL_CONFIG,
        z_crop=effective_refinement_z_crop,
        postfilters=REFINED_MARKER_POSTFILTERS,
        min_intensity_measure=REFINED_MARKER_MIN_INTENSITY_MEASURE,
        min_intensity_threshold=REFINED_MARKER_MIN_INTENSITY_THRESHOLD,
        local_contrast_k=REFINED_MARKER_LOCAL_CONTRAST_K,
        local_contrast_shell_inner_radius=REFINED_MARKER_LOCAL_CONTRAST_SHELL_INNER_RADIUS,
        local_contrast_shell_outer_radius=REFINED_MARKER_LOCAL_CONTRAST_SHELL_OUTER_RADIUS,
        bright_pixel_measure=REFINED_MARKER_BRIGHT_PIXEL_MEASURE,
        bright_pixel_threshold=REFINED_MARKER_BRIGHT_PIXEL_THRESHOLD,
        bright_pixel_min_count=REFINED_MARKER_BRIGHT_PIXEL_MIN_COUNT,
        bright_pixel_min_fraction=REFINED_MARKER_BRIGHT_PIXEL_MIN_FRACTION)

    run_result = refine_run_result_from_cellpose_cache(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        colocalization_config=      COLOCALIZATION_CONFIG,
        cell_model_config=          refined_cell_model_config,
        marker_model_config=        None,#refined_marker_model_config,
        cell_cellprob_threshold=    REFINED_CELL_CELLPROB_THRESHOLD,
        cell_flow_threshold=        REFINED_CELL_FLOW_THRESHOLD,
        marker_cellprob_threshold=  REFINED_MARKER_CELLPROB_THRESHOLD,
        marker_flow_threshold=      REFINED_MARKER_FLOW_THRESHOLD,
        optional_region_result=     None)

    print("Refinement finished. "
          f"Overview rows: {len(run_result.tables.overview)}, "
          f"summary rows: {len(run_result.tables.summary)}.")

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
            show_optional_region_image=True)

        print(f"Inspecting refined visualization for:\n{SELECTED_FILE_NAME}")
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
    cell_masks_from_viewer, marker_masks_from_viewer = extract_label_masks_from_viewer(result_viewer)
    run_result = analyze_existing_masks(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        cell_masks=cell_masks_from_viewer,
        marker_masks=marker_masks_from_viewer,
        colocalization_config=COLOCALIZATION_CONFIG,
        optional_region_result=None,
        cell_refinement_context=run_result.cell_refinement_context,
        marker_refinement_context=run_result.marker_refinement_context,
        analysis_z_bounds=run_result.analysis_z_bounds,
    )

    print(run_result.tables.overview.iloc[:, :3])

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
    print("Inspect the relabeled result in napari and close the window when finished.")
    napari.run()
else:
    print("Manual label reanalysis from the napari viewer is disabled for this run.")
# %% EXPORT RESULTS
export_analysis_outputs(
    run_result=run_result,
    paths=loaded_images.paths,
    optional_region_result=None,
)
print("Final results exported.")

# %% END
