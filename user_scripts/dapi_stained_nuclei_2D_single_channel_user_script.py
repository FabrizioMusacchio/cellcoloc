"""Interactive single-channel segmentation demo for DAPI-stained nuclei.

This script demonstrates CellColoc's optional single-channel workflow on the
true 2D OME-TIFF dataset stored in ``example_data/dapi_stained_nuclei_2D``.
The dataset contains two channels. In this example, we segment and count the
DAPI-like nuclei channel only, *without* any colocalization step.

The workflow remains interactive and notebook-like:

1. choose and inspect the input file,
2. optionally analyze the whole image as one ROI or draw custom ROIs in napari,
3. optionally reuse a previously saved ROI mask from the results directory,
4. run one-channel segmentation and object counting within the ROI set,
5. inspect and optionally refine the resulting masks and tables.

author: Fabrizio Musacchio
date:   June 2026
"""
# %% IMPORTS
from dataclasses import replace
from pathlib import Path

import napari
import numpy as np

from cellcoloc import (
    CellposeModelConfig,
    RuntimeConfig,
    SingleChannelAnalysisConfig,
    SingleChannelConfig,
    SingleChannelDisplayNames,
    analyze_existing_single_channel_masks,
    create_full_image_roi_labels,
    create_single_channel_roi_drawing_viewer,
    export_single_channel_outputs,
    extract_single_channel_masks_from_viewer,
    load_roi_labels,
    load_single_channel_image,
    prepare_loaded_single_channel_image_for_analysis,
    refine_single_channel_run_result_from_cellpose_cache,
    run_roi_single_channel_segmentation,
    save_roi_labels_from_shapes,
    show_single_channel_results,
    try_load_single_channel_roi_labels)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# %% PROJECT SETTINGS
DATA_PATH = PROJECT_ROOT / "example_data" / "dapi_stained_nuclei_2D" / "dapi_stained_nuclei_2D.ome.tif"

CHANNEL_CONFIG = SingleChannelConfig(
    channel_index=1)

DISPLAY_NAMES = SingleChannelDisplayNames(
    channel="DAPI",
    objects="Segmented DAPI nuclei")

VOXEL_SCALE_ZYX = (0.325, 0.325)

MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",
    segmentation_method="cellpose",
    diameter=None,
    do_3d=None,
    z_crop=None,
    z_projection=None,
    anisotropy=False,
    flow3d_smooth=0,
    prefilter=None,
    postfilters=None,
    cellprob_threshold=0.0,
    flow_threshold=0.4)

ANALYSIS_CONFIG = SingleChannelAnalysisConfig(
    min_object_voxels=50)

RUNTIME_CONFIG = RuntimeConfig(
    draw_rois=True,
    process_rois=True,
    open_results=True,
    use_gpu=True,
    crop_for_testing=None,
    image_loading_mode="memory")

USE_FULL_IMAGE_AS_SINGLE_ROI = True
REUSE_EXISTING_ROI_MASK_IF_AVAILABLE = True
INITIAL_RESULT_LAYER_KEYS = [
    "channel_image",
    "rois",
    "roi_numbers",
    "masks"]
REFINEMENT_RESULT_LAYER_KEYS = [
    "masks",
    "rois"]

existing_roi_labels = None
result_viewer = None
# %% LOAD THE ANALYSIS CHANNEL
loaded_image = load_single_channel_image(
    source_path=DATA_PATH,
    channel_config=CHANNEL_CONFIG,
    voxel_scale_zyx=VOXEL_SCALE_ZYX,
    crop_for_testing=RUNTIME_CONFIG.crop_for_testing,
    image_loading_mode=RUNTIME_CONFIG.image_loading_mode)
loaded_image = prepare_loaded_single_channel_image_for_analysis(
    loaded_image,
    MODEL_CONFIG)
print(f"Results directory:\n{loaded_image.paths.results_dir}")

if REUSE_EXISTING_ROI_MASK_IF_AVAILABLE:
    existing_roi_labels = try_load_single_channel_roi_labels(loaded_image.paths.roi_mask_path)
# %% DRAW ROIS INTERACTIVELY IN NAPARI
if USE_FULL_IMAGE_AS_SINGLE_ROI:
    print("Whole-image mode is enabled. ROI drawing is skipped.")
elif existing_roi_labels is not None:
    print("An existing ROI mask was found and will be reused. ROI drawing is skipped.")
elif RUNTIME_CONFIG.draw_rois:
    roi_viewer, shapes_layer = create_single_channel_roi_drawing_viewer(
        loaded_image=loaded_image,
        display_names=DISPLAY_NAMES)
    print("Draw ROIs in napari and close the window. Then run the next cell.")
    napari.run()
else:
    print("ROI drawing is disabled. The next cell will load an existing ROI mask from disk.")
# %% SAVE THE DRAWN ROIS OR LOAD AN EXISTING ROI MASK
if USE_FULL_IMAGE_AS_SINGLE_ROI:
    roi_labels_2d = create_full_image_roi_labels(loaded_image.image.shape[1:])
elif existing_roi_labels is not None:
    roi_labels_2d = existing_roi_labels
elif RUNTIME_CONFIG.draw_rois:
    roi_labels_2d = save_roi_labels_from_shapes(
        shapes_layer=shapes_layer,
        output_path=loaded_image.paths.roi_mask_path,
        image_shape_yx=loaded_image.image.max(axis=0).shape,
        scale_yx=loaded_image.voxel_scale_zyx[1:])
else:
    roi_labels_2d = load_roi_labels(loaded_image.paths.roi_mask_path)

roi_ids = np.unique(roi_labels_2d)
roi_ids = roi_ids[roi_ids != 0]
print(f"ROI ids: {roi_ids}")
# %% RUN THE ROI-WISE SINGLE-CHANNEL SEGMENTATION AND COUNTING
run_result = run_roi_single_channel_segmentation(
    loaded_image=loaded_image,
    roi_labels_2d=roi_labels_2d,
    model_config=MODEL_CONFIG,
    analysis_config=ANALYSIS_CONFIG,
    runtime_config=RUNTIME_CONFIG,)
print("Single-channel analysis finished. "
      f"Overview rows: {len(run_result.tables.overview)}, "
      f"object rows: {len(run_result.tables.objects)}.")
# %% VISUALIZE THE RESULT IN NAPARI
if RUNTIME_CONFIG.open_results:
    result_viewer = show_single_channel_results(
        loaded_image=loaded_image,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        display_names=DISPLAY_NAMES,
        viewer=result_viewer,
        layers_to_show=INITIAL_RESULT_LAYER_KEYS,
        replace_existing_layers=True)
    print("Inspect the single-channel result in napari and close the window when finished.")
    napari.run()
# %% OPTIONALLY REFINE RESULTS AND VISUALIZE UPDATED RESULT IN NAPARI
REFINE_WITH_CACHED_CELLPOSE_OUTPUTS = False
REFINEMENT_ANALYSIS_Z_CROP = None

REFINED_CELLPROB_THRESHOLD = MODEL_CONFIG.cellprob_threshold - 2.0
REFINED_FLOW_THRESHOLD = MODEL_CONFIG.flow_threshold

REFINED_POSTFILTERS = None
REFINED_MIN_INTENSITY_MEASURE = "mean"
REFINED_MIN_INTENSITY_THRESHOLD = None
REFINED_LOCAL_CONTRAST_K = 1.0
REFINED_LOCAL_CONTRAST_SHELL_INNER_RADIUS = 1.0
REFINED_LOCAL_CONTRAST_SHELL_OUTER_RADIUS = 4
REFINED_BRIGHT_PIXEL_MEASURE = "count"
REFINED_BRIGHT_PIXEL_THRESHOLD = None
REFINED_BRIGHT_PIXEL_MIN_COUNT = None
REFINED_BRIGHT_PIXEL_MIN_FRACTION = None

effective_refinement_z_crop = (
    MODEL_CONFIG.z_crop if REFINEMENT_ANALYSIS_Z_CROP is None else REFINEMENT_ANALYSIS_Z_CROP)

if REFINE_WITH_CACHED_CELLPOSE_OUTPUTS:
    refined_model_config = replace(
        MODEL_CONFIG,
        z_crop=effective_refinement_z_crop,
        postfilters=REFINED_POSTFILTERS,
        min_intensity_measure=REFINED_MIN_INTENSITY_MEASURE,
        min_intensity_threshold=REFINED_MIN_INTENSITY_THRESHOLD,
        local_contrast_k=REFINED_LOCAL_CONTRAST_K,
        local_contrast_shell_inner_radius=REFINED_LOCAL_CONTRAST_SHELL_INNER_RADIUS,
        local_contrast_shell_outer_radius=REFINED_LOCAL_CONTRAST_SHELL_OUTER_RADIUS,
        bright_pixel_measure=REFINED_BRIGHT_PIXEL_MEASURE,
        bright_pixel_threshold=REFINED_BRIGHT_PIXEL_THRESHOLD,
        bright_pixel_min_count=REFINED_BRIGHT_PIXEL_MIN_COUNT,
        bright_pixel_min_fraction=REFINED_BRIGHT_PIXEL_MIN_FRACTION)

    run_result = refine_single_channel_run_result_from_cellpose_cache(
        loaded_image=loaded_image,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        analysis_config=ANALYSIS_CONFIG,
        model_config=refined_model_config,
        cellprob_threshold=REFINED_CELLPROB_THRESHOLD,
        flow_threshold=REFINED_FLOW_THRESHOLD)

    print("Refined single-channel analysis finished. "
          f"Overview rows: {len(run_result.tables.overview)}, "
          f"object rows: {len(run_result.tables.objects)}.")

    result_viewer = show_single_channel_results(
        loaded_image=loaded_image,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        display_names=DISPLAY_NAMES,
        viewer=result_viewer,
        layers_to_show=REFINEMENT_RESULT_LAYER_KEYS,
        replace_existing_layers=True)
    print("Inspect the refined single-channel result in napari and close the window when finished.")
    napari.run()
else:
    print("Cached Cellpose refinement is disabled for this run.")

# %% OPTIONALLY REANALYZE MANUALLY EDITED LABEL LAYERS FROM NAPARI
REANALYZE_EDITED_LABELS_FROM_VIEWER = False

if REANALYZE_EDITED_LABELS_FROM_VIEWER:
    masks_from_viewer = extract_single_channel_masks_from_viewer(
        result_viewer,
        object_layer_name=DISPLAY_NAMES.objects)
    run_result = analyze_existing_single_channel_masks(
        loaded_image=loaded_image,
        roi_labels_2d=roi_labels_2d,
        masks=masks_from_viewer,
        analysis_config=ANALYSIS_CONFIG,
        analysis_z_bounds=run_result.analysis_z_bounds,
        refinement_context=run_result.refinement_context)

    result_viewer = show_single_channel_results(
        loaded_image=loaded_image,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        display_names=DISPLAY_NAMES,
        viewer=result_viewer,
        layers_to_show=REFINEMENT_RESULT_LAYER_KEYS,
        replace_existing_layers=True)
    print("Inspect the relabeled single-channel result in napari and close the window when finished.")
    napari.run()
else:
    print("Manual label reanalysis from the napari viewer is disabled for this run.")
# %% EXPORT RESULTS
export_single_channel_outputs(
    run_result=run_result,
    paths=loaded_image.paths)
print("Final single-channel results exported.")
# %% END
