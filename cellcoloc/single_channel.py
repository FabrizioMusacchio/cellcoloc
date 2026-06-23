"""Single-channel segmentation workflow built on top of CellColoc core tools.

This module provides a parallel high-level API for workflows that only need to
segment and count objects in one image channel. It reuses the same OMIO
loading, ROI handling, pre- and postfiltering, z-cropping, z projection,
Cellpose or threshold segmentation, and cached Cellpose refinement backends as
the multi-channel colocalization pipeline.

author: Fabrizio Musacchio
date: June 2026
"""
# %% IMPORTS
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import omio as om
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

from .analysis import (
    _apply_analysis_z_bounds,
    _compute_mask_occupancy_metrics,
    _normalize_z_crop_bounds,
    _project_zyx_volume,
    _resolve_analysis_z_bounds,
    _resolve_analysis_z_projection_method,
)
from .config import (
    CellposeModelConfig,
    RuntimeConfig,
    SingleChannelAnalysisConfig,
    SingleChannelConfig,
    SingleChannelDisplayNames,
)
from .filtering import apply_postfilters, apply_prefilter
from .io import _extract_zyx_channel, _resolve_voxel_scale_zyx, load_roi_labels, save_roi_labels
from .roi import get_bbox_2d, get_roi_label_points
from .schemas import (
    CellposeChannelRefinementContext,
    CellposeRefinementRoiCache,
    LoadedSingleChannelImage,
    SingleChannelResultsPaths,
    SingleChannelRunResult,
    SingleChannelTables,
)
from .segmentation import (
    create_cellpose_model,
    evaluate_segmentation_method,
    filter_labels_by_size,
    normalize_segmentation_method,
    relabel_with_offset,
)
from .visualization import (
    _build_roi_labels_3d,
    _get_or_create_viewer,
    _hide_layer_if_present,
    _normalize_layer_selection,
    _replace_or_add_image,
    _replace_or_add_labels,
    _replace_or_add_points,
    _should_render_layer,
)

# %% IO HELPERS
def build_single_channel_results_paths(source_path: Path) -> SingleChannelResultsPaths:
    """Create the standard results paths for one single-channel dataset run."""

    source_path = Path(source_path).expanduser().resolve()
    results_dir = source_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    stem = source_path.stem
    return SingleChannelResultsPaths(
        source_path=source_path,
        results_dir=results_dir,
        roi_mask_path=results_dir / f"{stem}_roi_labelmask.tif",
        object_csv_path=results_dir / f"{stem}_single_channel_objects.csv",
        excel_path=results_dir / f"{stem}_single_channel_segmentation.xlsx",
        mask_path=results_dir / f"{stem}_single_channel_masks.tif",
    )


def load_single_channel_image(
    source_path: Path,
    channel_config: SingleChannelConfig,
    voxel_scale_zyx: tuple[float, float] | tuple[float, float, float] | None,
    crop_for_testing: tuple[slice, slice, slice] | None = None,
    image_loading_mode: str = "memory",
) -> LoadedSingleChannelImage:
    """Load one configured analysis channel from a microscopy dataset.

    Parameters
    ----------
    source_path:
        Input microscopy dataset that OMIO can open.
    channel_config:
        One-channel mapping defining which raw channel should be analyzed.
    voxel_scale_zyx:
        Optional explicit voxel size in micrometers, either as ``(Z, Y, X)``
        or, for 2D convenience, as ``(Y, X)``.
    crop_for_testing:
        Optional test crop applied after channel extraction in ``(Z, Y, X)``
        order.
    image_loading_mode:
        Raw-image loading strategy. ``"memory"`` materializes the full image
        eagerly, whereas ``"memap"`` keeps OMIO's disk-backed Zarr cache.
    """

    paths = build_single_channel_results_paths(source_path)
    normalized_loading_mode = image_loading_mode.strip().lower()
    if normalized_loading_mode == "memory":
        image_tzcyx, metadata = om.imread(paths.source_path, zarr_store=None)
        image_tzcyx = np.asarray(image_tzcyx)
    elif normalized_loading_mode == "memap":
        image_tzcyx, metadata = om.imread(
            paths.source_path,
            zarr_store="disk",
            reuse_disk_cache=True,
        )
    else:
        raise ValueError(
            "`image_loading_mode` must be 'memory' or 'memap', got "
            f"{image_loading_mode!r}."
        )

    print(f"Image loading mode: {normalized_loading_mode}")
    raw_z_size = int(image_tzcyx.shape[1])
    is_3d = raw_z_size > 1
    print(f"Loaded image: {paths.source_path}")
    print(f"Raw image shape (expected TZCYX): {image_tzcyx.shape}")
    print(f"Detected dimensionality from Z axis: {'3D' if is_3d else '2D'} (Z={raw_z_size})")

    resolved_voxel_scale_zyx = _resolve_voxel_scale_zyx(voxel_scale_zyx, metadata)
    image = _extract_zyx_channel(image_tzcyx, channel_config.channel_index)
    if crop_for_testing is not None:
        image = image[crop_for_testing]
    print(f"Analysis volume shape (ZYX): {image.shape}")

    return LoadedSingleChannelImage(
        source_path=paths.source_path,
        paths=paths,
        voxel_scale_zyx=resolved_voxel_scale_zyx,
        image=image,
        raw_shape_tzcyx=tuple(image_tzcyx.shape),
        raw_z_size=raw_z_size,
        is_3d=is_3d,
        metadata=metadata,
        analysis_z_bounds=None,
        z_projection_method=None,
    )


def try_load_single_channel_roi_labels(path: Path) -> np.ndarray | None:
    """Load a saved ROI label mask for a single-channel workflow when present."""

    path = Path(path)
    if not path.exists():
        print(f"No existing ROI label mask found at:\n{path}")
        return None
    return load_roi_labels(path)


def export_single_channel_outputs(
    run_result: SingleChannelRunResult,
    paths: SingleChannelResultsPaths,
) -> None:
    """Write standard tables and masks for one completed single-channel run."""

    run_result.tables.objects.to_csv(paths.object_csv_path, index=False)
    tifffile.imwrite(paths.mask_path, run_result.masks.astype(np.uint32))

    with pd.ExcelWriter(paths.excel_path) as writer:
        run_result.tables.objects.to_excel(writer, sheet_name="object_summary", index=False)
        run_result.tables.overview.to_excel(writer, sheet_name="roi_overview", index=False)

    print(f"Saved object CSV analysis to:\n{paths.object_csv_path}")
    print(f"Saved Excel analysis to:\n{paths.excel_path}")
    print(f"Saved segmentation masks to:\n{paths.mask_path}")


# %% ANALYSIS PREPARATION
def prepare_loaded_single_channel_image_for_analysis(
    loaded_image: LoadedSingleChannelImage,
    model_config: CellposeModelConfig | None,
) -> LoadedSingleChannelImage:
    """Prepare one loaded channel for downstream analysis according to config.

    When ``model_config.z_projection`` is set, the helper optionally applies
    the corresponding global ``z_crop`` first, then projects the image along
    z, and returns a singleton-z 2D analysis view.
    """

    projection_method = _resolve_analysis_z_projection_method(model_config)
    if projection_method is None:
        return loaded_image

    analysis_z_bounds = _resolve_analysis_z_bounds(
        loaded_image.image.shape[0],
        model_config,
    )
    z_slice = slice(*analysis_z_bounds) if analysis_z_bounds is not None else slice(None)
    projected_image = _project_zyx_volume(loaded_image.image[z_slice], projection_method)

    return LoadedSingleChannelImage(
        source_path=loaded_image.source_path,
        paths=loaded_image.paths,
        voxel_scale_zyx=(1.0, loaded_image.voxel_scale_zyx[1], loaded_image.voxel_scale_zyx[2]),
        image=projected_image,
        raw_shape_tzcyx=loaded_image.raw_shape_tzcyx,
        raw_z_size=loaded_image.raw_z_size,
        is_3d=False,
        metadata=loaded_image.metadata,
        analysis_z_bounds=analysis_z_bounds,
        z_projection_method=projection_method,
    )


# %% TABLE BUILDERS
def _build_single_channel_object_table(
    masks: np.ndarray,
    roi_labels_2d: np.ndarray,
) -> pd.DataFrame:
    """Create one object-summary row per segmented label."""

    if np.max(masks) == 0:
        return pd.DataFrame(
            columns=[
                "roi_id",
                "object_label",
                "object_voxels",
                "object_voxels_props",
                "centroid_z",
                "centroid_y",
                "centroid_x",
                "object_voxels_delta",
            ]
        )

    props_table = pd.DataFrame(
        regionprops_table(masks, properties=("label", "area", "centroid"))
    ).rename(
        columns={
            "label": "object_label",
            "area": "object_voxels_props",
            "centroid-0": "centroid_z",
            "centroid-1": "centroid_y",
            "centroid-2": "centroid_x",
        }
    )
    props_table["object_label"] = props_table["object_label"].astype(int)

    rows: list[dict[str, int | float]] = []
    roi_labels_3d = np.broadcast_to(roi_labels_2d, masks.shape)
    for object_label in props_table["object_label"]:
        object_mask = masks == object_label
        object_voxels = int(object_mask.sum())
        roi_values = roi_labels_3d[object_mask]
        roi_values = roi_values[roi_values != 0]
        roi_id = int(np.unique(roi_values)[0]) if roi_values.size > 0 else 0
        rows.append(
            {
                "roi_id": roi_id,
                "object_label": int(object_label),
                "object_voxels": object_voxels,
            }
        )

    object_table = pd.DataFrame(rows).merge(props_table, on="object_label", how="left")
    object_table["object_voxels_delta"] = (
        object_table["object_voxels"] - object_table["object_voxels_props"]
    )
    return object_table.sort_values(by=["roi_id", "object_label"]).reset_index(drop=True)


def _build_single_channel_overview_table(
    roi_labels_2d: np.ndarray,
    loaded_image: LoadedSingleChannelImage,
    masks: np.ndarray,
    object_table: pd.DataFrame,
    analysis_z_bounds: tuple[int, int] | None,
) -> pd.DataFrame:
    """Create one ROI overview row per ROI for one-channel analyses."""

    z_size_um, y_size_um, x_size_um = loaded_image.voxel_scale_zyx
    pixel_area_um2 = y_size_um * x_size_um
    voxel_volume_um3 = z_size_um * y_size_um * x_size_um
    n_z = loaded_image.image.shape[0]
    z_start, z_stop = analysis_z_bounds if analysis_z_bounds is not None else (0, n_z)
    analysis_depth = z_stop - z_start

    rows: list[dict[str, int | float]] = []
    for roi_id in np.unique(roi_labels_2d):
        if roi_id == 0:
            continue

        roi_mask_2d = roi_labels_2d == roi_id
        roi_area_px = int(roi_mask_2d.sum())
        roi_area_um2 = float(roi_area_px * pixel_area_um2)
        roi_volume_voxels = int(roi_area_px * analysis_depth)
        roi_volume_um3 = float(roi_volume_voxels * voxel_volume_um3)
        object_rows = object_table[object_table["roi_id"] == roi_id]

        row: dict[str, int | float] = {
            "roi_id": int(roi_id),
            "n_objects": int(len(object_rows)),
            "drawn_roi_area_px": roi_area_px,
            "drawn_roi_area_um2": roi_area_um2,
            "roi_volume_voxels": roi_volume_voxels,
            "roi_volume_um3": roi_volume_um3,
        }
        row.update(
            _compute_mask_occupancy_metrics(
                "object",
                masks,
                roi_mask_2d,
                loaded_image.voxel_scale_zyx,
                analysis_z_bounds,
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


# %% ANALYSIS CORE
def analyze_existing_single_channel_masks(
    loaded_image: LoadedSingleChannelImage,
    roi_labels_2d: np.ndarray,
    masks: np.ndarray,
    analysis_config: SingleChannelAnalysisConfig,
    analysis_z_bounds: tuple[int, int] | None = None,
    refinement_context: CellposeChannelRefinementContext | None = None,
    model_config: CellposeModelConfig | None = None,
) -> SingleChannelRunResult:
    """Recompute object tables from existing one-channel label masks."""

    effective_analysis_z_bounds = (
        None if loaded_image.z_projection_method is not None else analysis_z_bounds
    )
    full_masks = _apply_analysis_z_bounds(masks, effective_analysis_z_bounds)

    print(f"\nFiltering objects smaller than {analysis_config.min_object_voxels} voxels...")
    full_masks = filter_labels_by_size(full_masks, analysis_config.min_object_voxels)

    if model_config is not None and model_config.postfilters is not None:
        print("Applying configured postfilters to single-channel masks...")
        full_masks = apply_postfilters(
            full_masks,
            loaded_image.image,
            model_config,
        )

    object_table = _build_single_channel_object_table(full_masks, roi_labels_2d)
    overview_table = _build_single_channel_overview_table(
        roi_labels_2d=roi_labels_2d,
        loaded_image=loaded_image,
        masks=full_masks,
        object_table=object_table,
        analysis_z_bounds=effective_analysis_z_bounds,
    )

    return SingleChannelRunResult(
        masks=full_masks,
        tables=SingleChannelTables(
            objects=object_table,
            overview=overview_table,
        ),
        analysis_z_bounds=effective_analysis_z_bounds,
        refinement_context=refinement_context,
    )


def run_roi_single_channel_segmentation(
    loaded_image: LoadedSingleChannelImage,
    roi_labels_2d: np.ndarray,
    model_config: CellposeModelConfig,
    analysis_config: SingleChannelAnalysisConfig,
    runtime_config: RuntimeConfig,
) -> SingleChannelRunResult:
    """Run ROI-wise segmentation and object counting for one analysis channel."""

    if not runtime_config.process_rois:
        raise ValueError("ROI processing is disabled in RuntimeConfig.")

    roi_ids = np.unique(roi_labels_2d)
    roi_ids = roi_ids[roi_ids != 0]
    print(f"Found {len(roi_ids)} ROIs: {roi_ids}")

    if loaded_image.z_projection_method is not None:
        analysis_z_bounds = None
    else:
        analysis_z_bounds = _resolve_analysis_z_bounds(
            loaded_image.image.shape[0],
            model_config,
        )
    z_slice = slice(*analysis_z_bounds) if analysis_z_bounds is not None else slice(None)

    method = normalize_segmentation_method(model_config.segmentation_method)
    model = None
    if method == "cellpose":
        model = create_cellpose_model(model_config.model_name_or_path, runtime_config.use_gpu)

    full_masks = np.zeros(loaded_image.image.shape, dtype=np.uint32)
    roi_caches: list[CellposeRefinementRoiCache] = []
    label_offset = 0

    for roi_id in roi_ids:
        print(f"\nProcessing ROI {int(roi_id)}...")
        roi_mask_2d = roi_labels_2d == roi_id
        bbox = get_bbox_2d(roi_mask_2d)
        if bbox is None:
            print(f"Skipping ROI {int(roi_id)}: empty ROI")
            continue

        y_slice, x_slice = bbox
        roi_mask_crop_2d = roi_mask_2d[y_slice, x_slice]
        image_crop = loaded_image.image[z_slice, y_slice, x_slice].copy()
        image_crop = apply_prefilter(image_crop, model_config)
        image_crop[:, ~roi_mask_crop_2d] = 0

        masks_roi, refinement_cache = evaluate_segmentation_method(
            model,
            image_crop,
            model_config,
            loaded_image.voxel_scale_zyx,
        )
        if refinement_cache is not None:
            refinement_cache.roi_id = int(roi_id)
            refinement_cache.y_min = int(y_slice.start)
            refinement_cache.y_max = int(y_slice.stop)
            refinement_cache.x_min = int(x_slice.start)
            refinement_cache.x_max = int(x_slice.stop)
            refinement_cache.roi_mask_crop_2d = roi_mask_crop_2d.copy()
            roi_caches.append(refinement_cache)

        masks_roi = relabel_with_offset(masks_roi, label_offset)
        if masks_roi.max() > 0:
            label_offset = int(masks_roi.max())

        full_masks[z_slice, y_slice, x_slice] = np.maximum(
            full_masks[z_slice, y_slice, x_slice],
            masks_roi,
        )

    refinement_context = None
    if roi_caches:
        refinement_context = CellposeChannelRefinementContext(
            model=model,
            model_name_or_path=model_config.model_name_or_path,
            roi_caches=roi_caches,
        )

    return analyze_existing_single_channel_masks(
        loaded_image=loaded_image,
        roi_labels_2d=roi_labels_2d,
        masks=full_masks,
        analysis_config=analysis_config,
        analysis_z_bounds=analysis_z_bounds,
        refinement_context=refinement_context,
        model_config=model_config,
    )


def _rebuild_single_channel_masks_from_refinement_context(
    image_shape: tuple[int, int, int],
    refinement_context: CellposeChannelRefinementContext,
    flow_threshold: float | None = None,
    cellprob_threshold: float | None = None,
) -> np.ndarray:
    """Recompute one-channel masks from cached Cellpose network outputs."""

    rebuilt_masks = np.zeros(image_shape, dtype=np.uint32)
    label_offset = 0

    for roi_cache in refinement_context.roi_caches:
        current_flow_threshold = roi_cache.flow_threshold if flow_threshold is None else flow_threshold
        current_cellprob_threshold = (
            roi_cache.cellprob_threshold if cellprob_threshold is None else cellprob_threshold
        )
        masks_roi = refinement_context.model._compute_masks(
            roi_cache.shape_for_masks,
            roi_cache.dP,
            roi_cache.cellprob,
            flow_threshold=current_flow_threshold,
            cellprob_threshold=current_cellprob_threshold,
            min_size=roi_cache.min_size,
            max_size_fraction=roi_cache.max_size_fraction,
            niter=roi_cache.niter,
            do_3D=roi_cache.do_3d,
            stitch_threshold=0.0,
        )
        masks_roi = np.asarray(masks_roi, dtype=np.uint32)
        if not roi_cache.do_3d:
            masks_roi = masks_roi[np.newaxis, :, :]

        masks_roi[:, ~roi_cache.roi_mask_crop_2d] = 0
        masks_roi = relabel_with_offset(masks_roi, label_offset)
        if masks_roi.max() > 0:
            label_offset = int(masks_roi.max())

        y_slice = slice(roi_cache.y_min, roi_cache.y_max)
        x_slice = slice(roi_cache.x_min, roi_cache.x_max)
        rebuilt_masks[:, y_slice, x_slice] = np.maximum(
            rebuilt_masks[:, y_slice, x_slice],
            masks_roi,
        )

    return rebuilt_masks


def refine_single_channel_run_result_from_cellpose_cache(
    loaded_image: LoadedSingleChannelImage,
    roi_labels_2d: np.ndarray,
    run_result: SingleChannelRunResult,
    analysis_config: SingleChannelAnalysisConfig,
    model_config: CellposeModelConfig | None = None,
    cellprob_threshold: float | None = None,
    flow_threshold: float | None = None,
) -> SingleChannelRunResult:
    """Recompute one-channel masks and tables from cached Cellpose outputs.

    Passing ``model_config=None`` leaves the current masks unchanged and simply
    rebuilds the object tables from the masks already stored in
    ``run_result``.
    """

    if loaded_image.z_projection_method is not None:
        analysis_z_bounds = None
    else:
        analysis_z_bounds = (
            run_result.analysis_z_bounds
            if model_config is None
            else _resolve_analysis_z_bounds(
                loaded_image.image.shape[0],
                model_config,
                fallback=run_result.analysis_z_bounds,
            )
        )

    if model_config is None:
        rebuilt_masks = np.asarray(run_result.masks, dtype=np.uint32).copy()
    else:
        if run_result.refinement_context is None:
            raise ValueError(
                "Single-channel refinement was requested, but this run result "
                "does not contain Cellpose refinement caches. Threshold-only "
                "refinement is currently available only when the initial "
                "segmentation was produced with a supported Cellpose 4 run."
            )
        rebuilt_masks = _rebuild_single_channel_masks_from_refinement_context(
            image_shape=loaded_image.image.shape,
            refinement_context=run_result.refinement_context,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

    rebuilt_masks = _apply_analysis_z_bounds(rebuilt_masks, analysis_z_bounds)
    return analyze_existing_single_channel_masks(
        loaded_image=loaded_image,
        roi_labels_2d=roi_labels_2d,
        masks=rebuilt_masks,
        analysis_config=analysis_config,
        analysis_z_bounds=analysis_z_bounds,
        refinement_context=run_result.refinement_context,
        model_config=model_config,
    )


# %% VISUALIZATION
def create_single_channel_roi_drawing_viewer(
    loaded_image: LoadedSingleChannelImage,
    display_names: SingleChannelDisplayNames | None = None,
):
    """Open a napari viewer for drawing 2D ROIs on one-channel projections."""

    import napari

    display_names = display_names or SingleChannelDisplayNames()
    projection = loaded_image.image.max(axis=0)

    viewer = napari.Viewer()
    viewer.add_image(
        projection,
        name=f"{display_names.channel} max projection for ROI drawing",
        scale=loaded_image.voxel_scale_zyx[1:],
    )
    shapes_layer = viewer.add_shapes(
        name="Draw ROIs here",
        ndim=2,
        shape_type="polygon",
        edge_width=2,
        face_color="transparent",
        blending="additive",
    )
    return viewer, shapes_layer


def extract_single_channel_masks_from_viewer(
    viewer,
    object_layer_name: str = "Segmented objects",
) -> np.ndarray:
    """Extract the current one-channel labels layer from a napari viewer."""

    if object_layer_name not in viewer.layers:
        raise KeyError(f"Object label layer not found in viewer: {object_layer_name}")
    return np.asarray(viewer.layers[object_layer_name].data, dtype=np.uint32)


def show_single_channel_results(
    loaded_image: LoadedSingleChannelImage,
    roi_labels_2d: np.ndarray,
    run_result: SingleChannelRunResult,
    display_names: SingleChannelDisplayNames | None = None,
    viewer=None,
    layers_to_show: Sequence[str] | None = None,
    replace_existing_layers: bool = True,
):
    """Display or refresh one-channel analysis layers in napari.

    Supported ``layers_to_show`` keys are ``"channel_image"``, ``"rois"``,
    ``"roi_numbers"``, and ``"masks"``.
    """

    display_names = display_names or SingleChannelDisplayNames()
    viewer = _get_or_create_viewer(viewer)
    selected_layers = _normalize_layer_selection(layers_to_show)

    if _should_render_layer(selected_layers, "channel_image"):
        _replace_or_add_image(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=loaded_image.image,
            name=display_names.channel,
            scale=loaded_image.voxel_scale_zyx,
            blending="additive",
            colormap="magenta",
            channel_axis=None,
        )

    roi_labels_3d = _build_roi_labels_3d(
        roi_labels_2d,
        loaded_image.image.shape[0],
        run_result.analysis_z_bounds,
    )
    if _should_render_layer(selected_layers, "rois"):
        _replace_or_add_labels(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=roi_labels_3d,
            name="ROIs",
            blending="additive",
            scale=loaded_image.voxel_scale_zyx,
        )

    roi_points_yx, roi_text_labels = get_roi_label_points(roi_labels_2d)
    if len(roi_points_yx) > 0 and _should_render_layer(selected_layers, "roi_numbers"):
        _replace_or_add_points(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=roi_points_yx,
            name="ROI numbers",
            scale=loaded_image.voxel_scale_zyx[1:],
            size=10,
            face_color="transparent",
            text={
                "string": roi_text_labels,
                "size": 14,
                "color": "white",
                "anchor": "center",
            },
        )

    if _should_render_layer(selected_layers, "masks"):
        _replace_or_add_labels(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=run_result.masks,
            name=display_names.objects,
            blending="additive",
            scale=loaded_image.voxel_scale_zyx,
        )

    _hide_layer_if_present(viewer, f"{display_names.channel} max projection for ROI drawing")
    _hide_layer_if_present(viewer, "Draw ROIs here")
    return viewer


__all__ = [
    "SingleChannelConfig",
    "SingleChannelDisplayNames",
    "SingleChannelAnalysisConfig",
    "SingleChannelResultsPaths",
    "LoadedSingleChannelImage",
    "SingleChannelTables",
    "SingleChannelRunResult",
    "build_single_channel_results_paths",
    "load_single_channel_image",
    "prepare_loaded_single_channel_image_for_analysis",
    "try_load_single_channel_roi_labels",
    "run_roi_single_channel_segmentation",
    "analyze_existing_single_channel_masks",
    "refine_single_channel_run_result_from_cellpose_cache",
    "create_single_channel_roi_drawing_viewer",
    "show_single_channel_results",
    "extract_single_channel_masks_from_viewer",
    "export_single_channel_outputs",
]
