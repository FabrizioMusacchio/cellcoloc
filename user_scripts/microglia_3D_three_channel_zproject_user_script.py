"""Interactive three-channel microglia demo with full-stack z projection.

This script demonstrates how a nominally 3D dataset can be analyzed through a
global z projection before segmentation. The full stack is projected over all
available z slices, after which CellColoc operates on the resulting 2D
analysis view.

Compared with a full 3D segmentation workflow, this can reduce computational
cost substantially and may be sufficient when the biological question does not
require true 3D object separation along z.

Biological channel mapping used here:

- channel 0: ``Cx3cr1-tdTomato`` microglia reporter signal,
- channel 1: ``Iba1`` staining,
- channel 2: ``DAPI`` segmentation demo channel.

The script demonstrates three different cell-positivity views at the end:

1. cells positive for channel 0 + channel 1,
2. cells positive for channel 0 + channel 2,
3. cells positive for channel 0 + channel 1 + channel 2.

author: Fabrizio Musacchio
date:   June 2026
"""
# %% IMPORTS
from pathlib import Path
from dataclasses import replace

import napari
import numpy as np

from cellcoloc import (
    CellposeModelConfig,
    ChannelConfig,
    ColocalizationConfig,
    DisplayNames,
    RuntimeConfig,
    analyze_existing_masks,
    build_positive_cell_mask,
    create_full_image_roi_labels,
    create_roi_drawing_viewer,
    export_analysis_outputs,
    extract_label_masks_from_viewer,
    load_analysis_images,
    load_roi_labels,
    prepare_loaded_images_for_analysis,
    refine_run_result_from_cellpose_cache,
    run_roi_cellpose_colocalization,
    save_roi_labels_from_shapes,
    show_analysis_results,
    try_load_roi_labels)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# %% PROJECT SETTINGS
DATA_DIR = PROJECT_ROOT / "example_data" / "microglia_3D"
DATA_PATHS = sorted(DATA_DIR.glob("*"))
allowed_extensions = [".czi", ".tif", ".tiff", ".ome.tif", ".ome.tiff"]
DATA_PATHS = [p for p in DATA_PATHS if p.suffix.lower() in allowed_extensions]
DATA_PATHS = [p for p in DATA_PATHS if not p.name.startswith(".")]

CHANNEL_CONFIG = ChannelConfig(
    cell_channel=0,
    marker_channel=1,
    optional_region_channel=2)

DISPLAY_NAMES = DisplayNames(
    cell="Cx3cr1-tdTomato",
    marker="Iba1",
    optional_region="DAPI",
    positive_cells="tdTomato + Iba1 positive masks")

VOXEL_SCALE_ZYX = None

CELL_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",
    segmentation_method="cellpose",
    diameter=None,
    z_crop=None,        # None; optional global analysis z crop as (start, stop); applies to all channels and all ROIs
    z_projection="max",  # None, "max", "mean", "median", "std", or "var"
    anisotropy=True,
    flow3d_smooth=0,
    prefilter=None,
    prefilter_sigma_xy=0.0,
    prefilter_sigma_z=0.0,
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
    cellprob_threshold=1.5,
    flow_threshold=0.4)

MARKER_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",
    segmentation_method="otsu",
    diameter=None,
    # z_crop=None,
    # z_projection="max", # it's not necessary to repeat z-projection option in marker channel; when set in the cell channel, it will be applied to all channels
    anisotropy=True,
    flow3d_smooth=0,
    prefilter=None,
    prefilter_sigma_xy=0.0,
    prefilter_sigma_z=0.0,
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
    cellprob_threshold=0.0,
    flow_threshold=0.4)

OPTIONAL_REGION_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",
    segmentation_method="cellpose",
    diameter=None,
    #z_crop=None,
    #z_projection="max", # it's not necessary to repeat z-projection option in optional region channel; when set in the cell channel, it will be applied to all channels
    anisotropy=True,
    flow3d_smooth=0,
    prefilter=None,
    prefilter_sigma_xy=0.0,
    prefilter_sigma_z=0.0,
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
    cellprob_threshold=0.0,
    flow_threshold=0.4)

"""
To exclude the optional third channel from segmentation and from the downstream
cell-positivity analysis, update the script as follows:

1. In ``CHANNEL_CONFIG``, set:

   ``optional_region_channel=None``

2. Set this variable to:

   ``OPTIONAL_REGION_MODEL_CONFIG = None``

3. In ``COLOCALIZATION_CONFIG``, set:

   ``evaluate_optional_region_cell_positivity=False``

4. Skip or remove the later visualization cells that explicitly show:

   - channel 0 + channel 2 positive cells
   - channel 0 + channel 1 + channel 2 positive cells

This reduces the workflow back to an ordinary two-channel analysis while
keeping the rest of the script structure intact.
"""

COLOCALIZATION_CONFIG = ColocalizationConfig(
    min_cell_voxels=50,
    overlap_fraction_threshold=0.02,
    min_overlap_voxels=10,
    evaluate_optional_region_cell_positivity=True)

RUNTIME_CONFIG = RuntimeConfig(
    draw_rois=True,
    process_rois=True,
    open_results=True,
    use_gpu=True,
    crop_for_testing=None,
    image_loading_mode="memap")

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
    "optional_region_labels"]
REFINEMENT_RESULT_LAYER_KEYS = [
    "cell_masks",
    "marker_masks",
    "positive_cells",
    "optional_region_labels",
    "rois"]

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
loaded_images = prepare_loaded_images_for_analysis(
    loaded_images,
    CELL_MODEL_CONFIG,
    MARKER_MODEL_CONFIG,
    OPTIONAL_REGION_MODEL_CONFIG)
print(f"Results directory:\n{loaded_images.paths.results_dir}")
print("Prepared analysis view: "
      f"shape={loaded_images.cell_image.shape}, "
      f"is_3d={loaded_images.is_3d}, "
      f"z_projection={loaded_images.z_projection_method!r}, "
      f"analysis_z_bounds={loaded_images.analysis_z_bounds}")

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
        display_names=DISPLAY_NAMES,
    )
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
# %% RUN THE ROI-WISE THREE-CHANNEL SEGMENTATION AND COLOCALIZATION ANALYSIS
run_result = run_roi_cellpose_colocalization(
    loaded_images=loaded_images,
    roi_labels_2d=roi_labels_2d,
    cell_model_config=CELL_MODEL_CONFIG,
    marker_model_config=MARKER_MODEL_CONFIG,
    colocalization_config=COLOCALIZATION_CONFIG,
    runtime_config=RUNTIME_CONFIG,
    optional_region_model_config=OPTIONAL_REGION_MODEL_CONFIG,
    optional_region_result=None)
print("Projected three-channel analysis finished. "
      f"Overview rows: {len(run_result.tables.overview)}, "
      f"summary rows: {len(run_result.tables.summary)}.")
# %% VISUALIZE THE BASE RESULT IN NAPARI
if RUNTIME_CONFIG.open_results:
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
# %% OPTIONALLY REFINE ALL THREE CHANNELS WITH CACHED CELLPOSE OUTPUTS
REFINE_WITH_CACHED_CELLPOSE_OUTPUTS = True
REFINEMENT_ANALYSIS_Z_CROP = None

REFINED_CELL_CELLPROB_THRESHOLD = CELL_MODEL_CONFIG.cellprob_threshold - 1.5
REFINED_CELL_FLOW_THRESHOLD     = CELL_MODEL_CONFIG.flow_threshold -0.1

REFINED_MARKER_CELLPROB_THRESHOLD = MARKER_MODEL_CONFIG.cellprob_threshold
REFINED_MARKER_FLOW_THRESHOLD = MARKER_MODEL_CONFIG.flow_threshold

REFINED_OPTIONAL_REGION_CELLPROB_THRESHOLD = OPTIONAL_REGION_MODEL_CONFIG.cellprob_threshold
REFINED_OPTIONAL_REGION_FLOW_THRESHOLD = OPTIONAL_REGION_MODEL_CONFIG.flow_threshold

REFINED_CELL_POSTFILTERS = None
REFINED_CELL_MIN_INTENSITY_MEASURE = "mean"
REFINED_CELL_MIN_INTENSITY_THRESHOLD = None
REFINED_CELL_LOCAL_CONTRAST_K = 1.0
REFINED_CELL_LOCAL_CONTRAST_SHELL_INNER_RADIUS = 1.0
REFINED_CELL_LOCAL_CONTRAST_SHELL_OUTER_RADIUS = 4
REFINED_CELL_BRIGHT_PIXEL_MEASURE = "count"
REFINED_CELL_BRIGHT_PIXEL_THRESHOLD = None
REFINED_CELL_BRIGHT_PIXEL_MIN_COUNT = None
REFINED_CELL_BRIGHT_PIXEL_MIN_FRACTION = None

REFINED_MARKER_POSTFILTERS = None
REFINED_MARKER_MIN_INTENSITY_MEASURE = "mean"
REFINED_MARKER_MIN_INTENSITY_THRESHOLD = None
REFINED_MARKER_LOCAL_CONTRAST_K = 1.0
REFINED_MARKER_LOCAL_CONTRAST_SHELL_INNER_RADIUS = 1.0
REFINED_MARKER_LOCAL_CONTRAST_SHELL_OUTER_RADIUS = 4
REFINED_MARKER_BRIGHT_PIXEL_MEASURE = "count"
REFINED_MARKER_BRIGHT_PIXEL_THRESHOLD = None
REFINED_MARKER_BRIGHT_PIXEL_MIN_COUNT = None
REFINED_MARKER_BRIGHT_PIXEL_MIN_FRACTION = None

REFINED_OPTIONAL_REGION_POSTFILTERS = None
REFINED_OPTIONAL_REGION_MIN_INTENSITY_MEASURE = "mean"
REFINED_OPTIONAL_REGION_MIN_INTENSITY_THRESHOLD = None
REFINED_OPTIONAL_REGION_LOCAL_CONTRAST_K = 1.0
REFINED_OPTIONAL_REGION_LOCAL_CONTRAST_SHELL_INNER_RADIUS = 1.0
REFINED_OPTIONAL_REGION_LOCAL_CONTRAST_SHELL_OUTER_RADIUS = 4
REFINED_OPTIONAL_REGION_BRIGHT_PIXEL_MEASURE = "count"
REFINED_OPTIONAL_REGION_BRIGHT_PIXEL_THRESHOLD = None
REFINED_OPTIONAL_REGION_BRIGHT_PIXEL_MIN_COUNT = None
REFINED_OPTIONAL_REGION_BRIGHT_PIXEL_MIN_FRACTION = None

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
    refined_optional_region_model_config = replace(
        OPTIONAL_REGION_MODEL_CONFIG,
        z_crop=effective_refinement_z_crop,
        postfilters=REFINED_OPTIONAL_REGION_POSTFILTERS,
        min_intensity_measure=REFINED_OPTIONAL_REGION_MIN_INTENSITY_MEASURE,
        min_intensity_threshold=REFINED_OPTIONAL_REGION_MIN_INTENSITY_THRESHOLD,
        local_contrast_k=REFINED_OPTIONAL_REGION_LOCAL_CONTRAST_K,
        local_contrast_shell_inner_radius=REFINED_OPTIONAL_REGION_LOCAL_CONTRAST_SHELL_INNER_RADIUS,
        local_contrast_shell_outer_radius=REFINED_OPTIONAL_REGION_LOCAL_CONTRAST_SHELL_OUTER_RADIUS,
        bright_pixel_measure=REFINED_OPTIONAL_REGION_BRIGHT_PIXEL_MEASURE,
        bright_pixel_threshold=REFINED_OPTIONAL_REGION_BRIGHT_PIXEL_THRESHOLD,
        bright_pixel_min_count=REFINED_OPTIONAL_REGION_BRIGHT_PIXEL_MIN_COUNT,
        bright_pixel_min_fraction=REFINED_OPTIONAL_REGION_BRIGHT_PIXEL_MIN_FRACTION)

    run_result = refine_run_result_from_cellpose_cache(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        run_result=run_result,
        colocalization_config=COLOCALIZATION_CONFIG,
        cell_model_config=refined_cell_model_config,
        marker_model_config=None,
        optional_region_model_config=refined_optional_region_model_config,
        cell_cellprob_threshold=REFINED_CELL_CELLPROB_THRESHOLD,
        cell_flow_threshold=REFINED_CELL_FLOW_THRESHOLD,
        marker_cellprob_threshold=REFINED_MARKER_CELLPROB_THRESHOLD,
        marker_flow_threshold=REFINED_MARKER_FLOW_THRESHOLD,
        optional_region_cellprob_threshold=REFINED_OPTIONAL_REGION_CELLPROB_THRESHOLD,
        optional_region_flow_threshold=REFINED_OPTIONAL_REGION_FLOW_THRESHOLD,
        optional_region_result=None)

    print("Projected refinement finished. "
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
REANALYZE_EDITED_LABELS_FROM_VIEWER = False

if REANALYZE_EDITED_LABELS_FROM_VIEWER:
    cell_masks_from_viewer, marker_masks_from_viewer = extract_label_masks_from_viewer(result_viewer)
    run_result = analyze_existing_masks(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        cell_masks=cell_masks_from_viewer,
        marker_masks=marker_masks_from_viewer,
        colocalization_config=COLOCALIZATION_CONFIG,
        optional_region_result=None,
        optional_region_masks=run_result.optional_region_masks,
        analysis_z_bounds=run_result.analysis_z_bounds,
        cell_refinement_context=run_result.cell_refinement_context,
        marker_refinement_context=run_result.marker_refinement_context,
        optional_region_refinement_context=run_result.optional_region_refinement_context)

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
    print("Inspect the relabeled result in napari and close the window when finished.")
    napari.run()
else:
    print("Manual label reanalysis from the napari viewer is disabled for this run.")

roi_display_labels = np.broadcast_to(roi_labels_2d, loaded_images.cell_image.shape).copy()
# %% VISUALIZE CELLS POSITIVE FOR CHANNEL 0 + CHANNEL 1
summary_channel01 = run_result.tables.summary.copy()
summary_channel01["marker_positive"] = summary_channel01["marker_positive"].astype(bool)
channel01_positive_masks = build_positive_cell_mask(run_result.cell_masks, summary_channel01)

viewer_01 = napari.Viewer()
viewer_01.add_image(
    loaded_images.cell_image,
    name=DISPLAY_NAMES.cell,
    scale=loaded_images.voxel_scale_zyx,
    blending="additive",
    colormap="magenta")
viewer_01.add_image(
    loaded_images.marker_image,
    name=DISPLAY_NAMES.marker,
    scale=loaded_images.voxel_scale_zyx,
    blending="additive",
    colormap="cyan")
viewer_01.add_labels(
    roi_display_labels,
    name="ROIs",
    scale=loaded_images.voxel_scale_zyx,
    opacity=0.35)
viewer_01.add_labels(
    channel01_positive_masks,
    name="Cells positive for channel 0 + channel 1",
    scale=loaded_images.voxel_scale_zyx,
    blending="additive")
print("Inspect channel-0 plus channel-1 positive cells.")
napari.run()
# %% VISUALIZE CELLS POSITIVE FOR CHANNEL 0 + CHANNEL 2
summary_channel02 = run_result.tables.summary.copy()
summary_channel02["marker_positive"] = summary_channel02["optional_region_positive"].astype(bool)
channel02_positive_masks = build_positive_cell_mask(run_result.cell_masks, summary_channel02)

viewer_02 = napari.Viewer()
viewer_02.add_image(
    loaded_images.cell_image,
    name=DISPLAY_NAMES.cell,
    scale=loaded_images.voxel_scale_zyx,
    blending="additive",
    colormap="magenta")
viewer_02.add_image(
    loaded_images.optional_region_image,
    name=DISPLAY_NAMES.optional_region,
    scale=loaded_images.voxel_scale_zyx,
    blending="additive",
    colormap="yellow")
viewer_02.add_labels(
    roi_display_labels,
    name="ROIs",
    scale=loaded_images.voxel_scale_zyx,
    opacity=0.35)
viewer_02.add_labels(
    channel02_positive_masks,
    name="Cells positive for channel 0 + channel 2",
    scale=loaded_images.voxel_scale_zyx,
    blending="additive")
print("Inspect channel-0 plus channel-2 positive cells.")
napari.run()
# %% VISUALIZE CELLS POSITIVE FOR CHANNEL 0 + CHANNEL 1 + CHANNEL 2
summary_channel012 = run_result.tables.summary.copy()
summary_channel012["marker_positive"] = summary_channel012["marker_and_optional_region_positive"].astype(bool)
channel012_positive_masks = build_positive_cell_mask(run_result.cell_masks, summary_channel012)

viewer_012 = napari.Viewer()
viewer_012.add_image(
    loaded_images.cell_image,
    name=DISPLAY_NAMES.cell,
    scale=loaded_images.voxel_scale_zyx,
    blending="additive",
    colormap="magenta")
viewer_012.add_image(
    loaded_images.marker_image,
    name=DISPLAY_NAMES.marker,
    scale=loaded_images.voxel_scale_zyx,
    blending="additive",
    colormap="cyan")
viewer_012.add_image(
    loaded_images.optional_region_image,
    name=DISPLAY_NAMES.optional_region,
    scale=loaded_images.voxel_scale_zyx,
    blending="additive",
    colormap="yellow")
viewer_012.add_labels(
    roi_display_labels,
    name="ROIs",
    scale=loaded_images.voxel_scale_zyx,
    opacity=0.35)
viewer_012.add_labels(
    channel012_positive_masks,
    name="Cells positive for channel 0 + channel 1 + channel 2",
    scale=loaded_images.voxel_scale_zyx,
    blending="additive")
print("Inspect channel-0 plus channel-1 plus channel-2 positive cells.")
napari.run()
# %% EXPORT RESULTS
export_analysis_outputs(
    run_result=run_result,
    paths=loaded_images.paths,
    optional_region_result=None)
print("Final results exported.")
# %% END
