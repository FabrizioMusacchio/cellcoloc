"""
Core Cellpose colocalization analysis logic.

author: Fabrizio Musacchio
date: May/June 2026
"""
# %% IMPORTS
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from .config import CellposeModelConfig, ColocalizationConfig, RuntimeConfig
from .filtering import apply_postfilters, apply_prefilter
from .roi import get_bbox_2d
from .schemas import (
    CellposeChannelRefinementContext,
    CellposeRefinementRoiCache,
    ColocalizationRunResult,
    ColocalizationTables,
    LoadedImageChannels,
    OptionalRegionSegmentationResult,
)
from .segmentation import (
    create_cellpose_model,
    create_cellpose_models_for_channels,
    evaluate_segmentation_method,
    filter_labels_by_size,
    normalize_segmentation_method,
    relabel_with_offset,
)

# %% MAIN ANALYSIS LOGIC
def analyze_label_overlaps(
    cell_masks: np.ndarray,
    marker_masks: np.ndarray,
    roi_id: int,
) -> list[dict[str, int | float]]:
    """Compute per-cell overlap rows against marker labels within one ROI."""

    rows: list[dict[str, int | float]] = []
    cell_labels = np.unique(cell_masks)
    cell_labels = cell_labels[cell_labels != 0]

    for cell_label in cell_labels:
        cell_mask = cell_masks == cell_label
        cell_voxels = int(cell_mask.sum())

        overlapping_markers = marker_masks[cell_mask]
        overlapping_markers = overlapping_markers[overlapping_markers != 0]
        unique_markers, counts = np.unique(overlapping_markers, return_counts=True)

        if len(unique_markers) == 0:
            rows.append(
                {
                    "roi_id": roi_id,
                    "cell_label": int(cell_label),
                    "cell_voxels": cell_voxels,
                    "n_overlapping_markers": 0,
                    "marker_label": np.nan,
                    "overlap_voxels": 0,
                    "overlap_fraction_of_cell": 0.0,
                }
            )
            continue

        for marker_label, overlap_voxels in zip(unique_markers, counts):
            rows.append(
                {
                    "roi_id": roi_id,
                    "cell_label": int(cell_label),
                    "cell_voxels": cell_voxels,
                    "n_overlapping_markers": int(len(unique_markers)),
                    "marker_label": int(marker_label),
                    "overlap_voxels": int(overlap_voxels),
                    "overlap_fraction_of_cell": float(overlap_voxels / cell_voxels),
                }
            )

    return rows


def build_positive_cell_mask(cell_masks: np.ndarray, summary_table: pd.DataFrame) -> np.ndarray:
    """Create a label image containing only marker-positive cells.

    The returned label image preserves the original cell labels for all cells
    classified as marker-positive in ``summary_table`` and sets every other
    voxel to zero.
    """

    if summary_table.empty:
        return np.zeros_like(cell_masks, dtype=np.uint32)

    positive_labels = summary_table.loc[summary_table["marker_positive"], "cell_label"].astype(np.uint32).to_numpy()
    max_label = int(cell_masks.max())
    positive_labels = positive_labels[positive_labels <= max_label]

    lookup = np.zeros(max_label + 1, dtype=np.uint32)
    lookup[positive_labels] = positive_labels
    return lookup[cell_masks]


def _normalize_z_crop_bounds(
    z_crop: tuple[int | None, int | None],
    z_size: int,
) -> tuple[int, int]:
    """Validate and normalize one user-supplied z crop against a stack size.

    Parameters
    ----------
    z_crop:
        Tuple of ``(start, stop)`` indices. ``None`` endpoints mean "from the
        start" or "to the end" respectively.
    z_size:
        Full z depth of the currently loaded stack.

    Returns
    -------
    tuple[int, int]
        Clipped and validated z bounds suitable for Python slicing.
    """

    start_raw, stop_raw = z_crop
    start = 0 if start_raw is None else int(start_raw)
    stop = z_size if stop_raw is None else int(stop_raw)

    start = max(0, min(start, z_size))
    stop = max(0, min(stop, z_size))
    if start >= stop:
        raise ValueError(
            "Invalid z crop bounds. Expected a tuple like ``(start, stop)`` "
            f"with start < stop after clipping to the stack size, got {z_crop!r} "
            f"for z size {z_size}."
        )

    return start, stop


def _resolve_analysis_z_bounds(
    z_size: int,
    *model_configs: CellposeModelConfig | None,
    fallback: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Resolve one global analysis z crop from one or more channel configs.

    The pipeline treats z cropping as a global analysis constraint. Individual
    channel configs may expose the same ``z_crop`` field for user convenience,
    but conflicting bounds across channels are rejected to keep all internal
    computations aligned.
    """

    normalized_bounds: list[tuple[int, int]] = []
    for model_config in model_configs:
        if model_config is None or model_config.z_crop is None:
            continue
        normalized_bounds.append(_normalize_z_crop_bounds(model_config.z_crop, z_size))

    if not normalized_bounds:
        return fallback

    first_bounds = normalized_bounds[0]
    if any(bounds != first_bounds for bounds in normalized_bounds[1:]):
        raise ValueError(
            "Conflicting z-crop bounds were provided across channel configs. "
            "Please use the same z crop for all participating channels."
        )

    return first_bounds


def _normalize_z_projection_method(z_projection: str | None) -> str | None:
    """Normalize and validate an optional global z-projection method.

    Supported methods are ``"max"``, ``"mean"``, ``"median"``, ``"std"``,
    and ``"var"``. ``None`` disables projection.
    """

    if z_projection is None:
        return None

    normalized = str(z_projection).strip().lower()
    allowed = {"max", "mean", "median", "std", "var"}
    if normalized not in allowed:
        raise ValueError(
            "`z_projection` must be one of None, 'max', 'mean', 'median', "
            f"'std', or 'var', got {z_projection!r}."
        )
    return normalized


def _resolve_analysis_z_projection_method(
    *model_configs: CellposeModelConfig | None,
) -> str | None:
    """Resolve one global z-projection method from one or more channel configs.

    The pipeline treats z projection as a global preprocessing choice. Channel
    configs may expose the same field for convenience, but conflicting methods
    across channels are rejected.
    """

    normalized_methods: list[str] = []
    for model_config in model_configs:
        if model_config is None:
            continue
        normalized_method = _normalize_z_projection_method(model_config.z_projection)
        if normalized_method is not None:
            normalized_methods.append(normalized_method)

    if not normalized_methods:
        return None

    first_method = normalized_methods[0]
    if any(method != first_method for method in normalized_methods[1:]):
        raise ValueError(
            "Conflicting z-projection methods were provided across channel "
            "configs. Please use the same z projection for all participating "
            "channels."
        )

    return first_method


def _project_zyx_volume(
    image_zyx: np.ndarray,
    projection_method: str,
) -> np.ndarray:
    """Project one ``ZYX`` image volume along z and keep singleton-z shape.

    Returns a float ``(1, Y, X)`` array so downstream code can continue to use
    the same ``ZYX`` interface even after a nominally 2D projection step.
    """

    image_float = np.asarray(image_zyx, dtype=np.float32)
    if projection_method == "max":
        projection_yx = np.max(image_float, axis=0)
    elif projection_method == "mean":
        projection_yx = np.mean(image_float, axis=0)
    elif projection_method == "median":
        projection_yx = np.median(image_float, axis=0)
    elif projection_method == "std":
        projection_yx = np.std(image_float, axis=0)
    elif projection_method == "var":
        projection_yx = np.var(image_float, axis=0)
    else:
        raise ValueError(f"Unsupported z projection method: {projection_method!r}.")

    return np.asarray(projection_yx, dtype=np.float32)[np.newaxis, :, :]


def prepare_loaded_images_for_analysis(
    loaded_images: LoadedImageChannels,
    *model_configs: CellposeModelConfig | None,
) -> LoadedImageChannels:
    """Prepare a loaded dataset for downstream analysis according to configs.

    This helper currently resolves an optional global z projection from the
    provided channel configs. When no projection method is configured, the
    original ``loaded_images`` object is returned unchanged. When a projection
    is requested, the helper optionally applies the globally configured z crop
    first, projects every available channel along z, and returns a new
    ``LoadedImageChannels`` bundle that behaves like a 2D dataset with
    singleton-z image arrays. All later ROI drawing, segmentation,
    quantification, and visualization steps should use this prepared bundle.

    Parameters
    ----------
    loaded_images:
        Previously loaded channel bundle from :func:`cell_coloc.io.load_analysis_images`.
    *model_configs:
        One or more participating channel configs. Any configured ``z_crop``
        and ``z_projection`` values are resolved globally across them.

    Returns
    -------
    LoadedImageChannels
        Either the original loaded image bundle or a projected analysis view.
    """

    projection_method = _resolve_analysis_z_projection_method(*model_configs)
    if projection_method is None:
        return loaded_images

    analysis_z_bounds = _resolve_analysis_z_bounds(
        loaded_images.cell_image.shape[0],
        *model_configs,
    )
    z_slice = slice(*analysis_z_bounds) if analysis_z_bounds is not None else slice(None)

    projected_cell_image = _project_zyx_volume(loaded_images.cell_image[z_slice], projection_method)
    projected_marker_image = _project_zyx_volume(loaded_images.marker_image[z_slice], projection_method)
    projected_optional_region_image = None
    if loaded_images.optional_region_image is not None:
        projected_optional_region_image = _project_zyx_volume(
            loaded_images.optional_region_image[z_slice],
            projection_method,
        )

    return LoadedImageChannels(
        source_path=loaded_images.source_path,
        paths=loaded_images.paths,
        voxel_scale_zyx=(1.0, loaded_images.voxel_scale_zyx[1], loaded_images.voxel_scale_zyx[2]),
        cell_image=projected_cell_image,
        marker_image=projected_marker_image,
        optional_region_image=projected_optional_region_image,
        raw_shape_tzcyx=loaded_images.raw_shape_tzcyx,
        raw_z_size=loaded_images.raw_z_size,
        is_3d=False,
        metadata=loaded_images.metadata,
        analysis_z_bounds=analysis_z_bounds,
        z_projection_method=projection_method,
    )


def _apply_analysis_z_bounds(
    label_image: np.ndarray | None,
    analysis_z_bounds: tuple[int, int] | None,
) -> np.ndarray | None:
    """Zero label content outside the active analysis z range.

    This helper keeps all mask arrays in full-stack shape for visualization and
    export, while ensuring that quantification and later refinement only see
    labels inside the chosen analysis z interval.
    """

    if label_image is None:
        return None
    if analysis_z_bounds is None:
        return np.asarray(label_image, dtype=np.uint32).copy()

    cropped = np.zeros_like(label_image, dtype=np.uint32)
    z_start, z_stop = analysis_z_bounds
    cropped[z_start:z_stop] = np.asarray(label_image[z_start:z_stop], dtype=np.uint32)
    return cropped


def analyze_existing_masks(
    loaded_images: LoadedImageChannels,
    roi_labels_2d: np.ndarray,
    cell_masks: np.ndarray,
    marker_masks: np.ndarray,
    colocalization_config: ColocalizationConfig,
    optional_region_result: OptionalRegionSegmentationResult | None = None,
    optional_region_masks: np.ndarray | None = None,
    analysis_z_bounds: tuple[int, int] | None = None,
    cell_refinement_context: CellposeChannelRefinementContext | None = None,
    marker_refinement_context: CellposeChannelRefinementContext | None = None,
    cell_model_config: CellposeModelConfig | None = None,
    marker_model_config: CellposeModelConfig | None = None,
) -> ColocalizationRunResult:
    """Recompute colocalization tables from existing label masks.

    This helper is used both after the initial Cellpose segmentation and after
    any later manual or threshold-based refinement of the label masks.

    Parameters
    ----------
    loaded_images:
        Loaded raw analysis channels and dataset metadata.
    roi_labels_2d:
        Drawn or generated 2D ROI label mask.
    cell_masks, marker_masks:
        Full-stack label masks for the two primary analysis channels. They may
        originate from Cellpose, thresholding, or manual relabeling.
    colocalization_config:
        Thresholds controlling how per-cell overlaps are interpreted.
    optional_region_result, optional_region_masks:
        Optional third-channel segmentation supplied either as the legacy
        result wrapper or directly as a label image. When both are provided,
        ``optional_region_masks`` takes precedence.
    analysis_z_bounds:
        Optional global z interval used for the current analysis. Labels
        outside this interval are ignored internally but the stored masks keep
        full-stack shape.
    cell_refinement_context, marker_refinement_context:
        Optional cached Cellpose network outputs used for later threshold-only
        refinement.
    cell_model_config, marker_model_config:
        Optional channel configs reused here mainly so postfilters can be
        applied consistently when masks are reanalyzed.

    Returns
    -------
    ColocalizationRunResult
        Structured masks and tables reflecting the provided segmentation state.
    """

    effective_analysis_z_bounds = (
        None if loaded_images.z_projection_method is not None else analysis_z_bounds
    )

    full_cell_masks = _apply_analysis_z_bounds(cell_masks, effective_analysis_z_bounds)
    full_marker_masks = _apply_analysis_z_bounds(marker_masks, effective_analysis_z_bounds)

    roi_ids = np.unique(roi_labels_2d)
    roi_ids = roi_ids[roi_ids != 0]

    print(f"\nFiltering cell labels smaller than {colocalization_config.min_cell_voxels} voxels...")
    full_cell_masks = filter_labels_by_size(full_cell_masks, colocalization_config.min_cell_voxels)

    if cell_model_config is not None and cell_model_config.postfilters is not None:
        print("Applying configured postfilters to cell masks...")
        full_cell_masks = apply_postfilters(
            full_cell_masks,
            loaded_images.cell_image,
            cell_model_config,
        )

    if marker_model_config is not None and marker_model_config.postfilters is not None:
        print("Applying configured postfilters to marker masks...")
        full_marker_masks = apply_postfilters(
            full_marker_masks,
            loaded_images.marker_image,
            marker_model_config,
        )

    detailed_rows: list[dict[str, int | float]] = []
    for roi_id in roi_ids:
        roi_mask_2d = roi_labels_2d == roi_id
        bbox = get_bbox_2d(roi_mask_2d)
        if bbox is None:
            continue

        y_slice, x_slice = bbox
        cell_roi = full_cell_masks[:, y_slice, x_slice]
        marker_roi = full_marker_masks[:, y_slice, x_slice]
        rows = analyze_label_overlaps(cell_roi, marker_roi, roi_id=int(roi_id))
        for row in rows:
            row["y_min"] = int(y_slice.start)
            row["y_max"] = int(y_slice.stop)
            row["x_min"] = int(x_slice.start)
            row["x_max"] = int(x_slice.stop)
        detailed_rows.extend(rows)

    detailed_table = pd.DataFrame(detailed_rows)
    if not detailed_table.empty:
        detailed_table = detailed_table.sort_values(
            by=["roi_id", "cell_label", "overlap_voxels"],
            ascending=[True, True, False],
        )

    effective_optional_region_masks = (
        np.asarray(optional_region_masks, dtype=np.uint32)
        if optional_region_masks is not None
        else (
            np.asarray(optional_region_result.labels, dtype=np.uint32)
            if optional_region_result is not None
            else None
        )
    )
    effective_optional_region_masks = _apply_analysis_z_bounds(
        effective_optional_region_masks,
        effective_analysis_z_bounds,
    )
    summary_table = _build_summary_table(
        detailed_table,
        full_cell_masks,
        colocalization_config,
        roi_labels_2d,
        effective_optional_region_masks,
    )

    overview_table = _build_overview_table(
        roi_labels_2d=roi_labels_2d,
        loaded_images=loaded_images,
        cell_masks=full_cell_masks,
        marker_masks=full_marker_masks,
        summary_table=summary_table,
        optional_region_masks=effective_optional_region_masks,
        analysis_z_bounds=effective_analysis_z_bounds,
    )
    positive_cell_masks = build_positive_cell_mask(full_cell_masks, summary_table)

    return ColocalizationRunResult(
        cell_masks=full_cell_masks,
        marker_masks=full_marker_masks,
        positive_cell_masks=positive_cell_masks,
        optional_region_masks=effective_optional_region_masks,
        analysis_z_bounds=effective_analysis_z_bounds,
        tables=ColocalizationTables(
            detailed=detailed_table,
            summary=summary_table,
            overview=overview_table,
        ),
        cell_refinement_context=cell_refinement_context,
        marker_refinement_context=marker_refinement_context,
    )


def _rebuild_masks_from_refinement_context(
    image_shape: tuple[int, int, int],
    refinement_context: CellposeChannelRefinementContext,
    flow_threshold: float | None = None,
    cellprob_threshold: float | None = None,
) -> np.ndarray:
    """Recompute full-size masks from cached Cellpose network outputs.

    The expensive neural-network forward pass is skipped here. Instead, this
    helper rebuilds masks only from stored Cellpose flow and cell-probability
    arrays for each ROI and stitches them back into one full-size label image.
    """

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


def refine_run_result_from_cellpose_cache(
    loaded_images: LoadedImageChannels,
    roi_labels_2d: np.ndarray,
    run_result: ColocalizationRunResult,
    colocalization_config: ColocalizationConfig,
    cell_model_config: CellposeModelConfig | None = None,
    marker_model_config: CellposeModelConfig | None = None,
    cell_cellprob_threshold: float | None = None,
    cell_flow_threshold: float | None = None,
    marker_cellprob_threshold: float | None = None,
    marker_flow_threshold: float | None = None,
    optional_region_result: OptionalRegionSegmentationResult | None = None,
) -> ColocalizationRunResult:
    """Recompute masks and tables from cached Cellpose outputs.

    This avoids rerunning the neural network forward pass and only recomputes
    the mask generation stage from cached ``dP`` and ``cellprob`` arrays.
    Passing ``cell_model_config=None`` and/or ``marker_model_config=None``
    leaves the respective channel unchanged and reuses the masks already stored
    in ``run_result``.

    Any z crop defined in the supplied refinement configs is interpreted as one
    global analysis z range and applied consistently across all channels. When
    no refinement config specifies a z crop, the function preserves the
    z-bounds stored in ``run_result``. When the loaded images already represent
    a z projection, additional z cropping is ignored because the data have
    already been collapsed to a singleton-z analysis view.
    """

    if loaded_images.z_projection_method is not None:
        analysis_z_bounds = None
    else:
        analysis_z_bounds = _resolve_analysis_z_bounds(
            loaded_images.cell_image.shape[0],
            cell_model_config,
            marker_model_config,
            fallback=run_result.analysis_z_bounds,
        )

    if cell_model_config is None:
        rebuilt_cell_masks = np.asarray(run_result.cell_masks, dtype=np.uint32).copy()
    else:
        if run_result.cell_refinement_context is None:
            raise ValueError(
                "Cell refinement was requested, but this run result does not "
                "contain Cellpose refinement caches for the cell channel. "
                "Threshold-only refinement is currently available only when "
                "the initial segmentation was produced with a supported "
                "Cellpose 4 run."
            )
        rebuilt_cell_masks = _rebuild_masks_from_refinement_context(
            image_shape=loaded_images.cell_image.shape,
            refinement_context=run_result.cell_refinement_context,
            flow_threshold=cell_flow_threshold,
            cellprob_threshold=cell_cellprob_threshold,
        )

    if marker_model_config is None:
        rebuilt_marker_masks = np.asarray(run_result.marker_masks, dtype=np.uint32).copy()
    else:
        if run_result.marker_refinement_context is None:
            raise ValueError(
                "Marker refinement was requested, but this run result does not "
                "contain Cellpose refinement caches for the marker channel. "
                "Threshold-only refinement is currently available only when "
                "the initial segmentation was produced with a supported "
                "Cellpose 4 run."
            )
        rebuilt_marker_masks = _rebuild_masks_from_refinement_context(
            image_shape=loaded_images.marker_image.shape,
            refinement_context=run_result.marker_refinement_context,
            flow_threshold=marker_flow_threshold,
            cellprob_threshold=marker_cellprob_threshold,
        )

    rebuilt_cell_masks = _apply_analysis_z_bounds(rebuilt_cell_masks, analysis_z_bounds)
    rebuilt_marker_masks = _apply_analysis_z_bounds(rebuilt_marker_masks, analysis_z_bounds)
    rebuilt_optional_region_masks = _apply_analysis_z_bounds(
        run_result.optional_region_masks,
        analysis_z_bounds,
    )

    return analyze_existing_masks(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        cell_masks=rebuilt_cell_masks,
        marker_masks=rebuilt_marker_masks,
        colocalization_config=colocalization_config,
        optional_region_result=optional_region_result,
        optional_region_masks=rebuilt_optional_region_masks,
        analysis_z_bounds=analysis_z_bounds,
        cell_refinement_context=run_result.cell_refinement_context,
        marker_refinement_context=run_result.marker_refinement_context,
        cell_model_config=cell_model_config,
        marker_model_config=marker_model_config,
    )


def _build_summary_table(
    detailed_table: pd.DataFrame,
    cell_masks: np.ndarray,
    config: ColocalizationConfig,
    roi_labels_2d: np.ndarray,
    optional_region_masks: np.ndarray | None = None,
) -> pd.DataFrame:
    """Aggregate detailed overlap rows into one summary row per cell.

    The summary retains the strongest marker overlap for each cell, classifies
    positivity according to ``ColocalizationConfig``, and optionally augments
    the result with third-channel positivity columns.
    """

    if detailed_table.empty:
        return pd.DataFrame(
            columns=[
                "roi_id",
                "cell_label",
                "cell_voxels",
                "marker_positive",
                "n_overlapping_markers",
                "best_marker_label",
                "best_overlap_voxels",
                "best_overlap_fraction",
                "cell_voxels_props",
                "centroid_z",
                "centroid_y",
                "centroid_x",
                "cell_voxels_delta",
            ]
        )

    cell_props = regionprops_table(cell_masks, properties=("label", "area", "centroid"))
    props_table = pd.DataFrame(cell_props).rename(
        columns={
            "label": "cell_label",
            "area": "cell_voxels_props",
            "centroid-0": "centroid_z",
            "centroid-1": "centroid_y",
            "centroid-2": "centroid_x",
        }
    )
    props_table["cell_label"] = props_table["cell_label"].astype(int)

    summary_rows: list[dict[str, int | float | bool]] = []
    for roi_id in np.unique(detailed_table["roi_id"]):
        detailed_roi = detailed_table[detailed_table["roi_id"] == roi_id]
        for cell_label in np.unique(detailed_roi["cell_label"]):
            detailed_cell = detailed_roi[detailed_roi["cell_label"] == cell_label]

            cell_voxels = int(detailed_cell["cell_voxels"].iloc[0])
            n_overlapping_markers = int(detailed_cell["n_overlapping_markers"].max())
            best_idx = detailed_cell["overlap_voxels"].idxmax()
            best_overlap_voxels = int(detailed_cell.loc[best_idx, "overlap_voxels"])
            best_overlap_fraction = float(detailed_cell.loc[best_idx, "overlap_fraction_of_cell"])
            best_marker_label = detailed_cell.loc[best_idx, "marker_label"]

            marker_positive = (
                (n_overlapping_markers > 0)
                and (best_overlap_voxels >= config.min_overlap_voxels)
                and (best_overlap_fraction >= config.overlap_fraction_threshold)
            )

            summary_rows.append(
                {
                    "roi_id": int(roi_id),
                    "cell_label": int(cell_label),
                    "cell_voxels": cell_voxels,
                    "marker_positive": bool(marker_positive),
                    "n_overlapping_markers": n_overlapping_markers,
                    "best_marker_label": int(best_marker_label) if not pd.isna(best_marker_label) else np.nan,
                    "best_overlap_voxels": best_overlap_voxels,
                    "best_overlap_fraction": best_overlap_fraction,
                }
            )

    summary_table = pd.DataFrame(summary_rows).merge(props_table, on="cell_label", how="left")
    summary_table["cell_voxels_delta"] = summary_table["cell_voxels"] - summary_table["cell_voxels_props"]

    if config.evaluate_optional_region_cell_positivity:
        optional_region_summary = _build_optional_region_summary_table(
            roi_labels_2d=roi_labels_2d,
            cell_masks=cell_masks,
            optional_region_masks=optional_region_masks,
            config=config,
        )
        summary_table = summary_table.merge(
            optional_region_summary,
            on=["roi_id", "cell_label"],
            how="left",
        )
        summary_table["optional_region_positive"] = summary_table["optional_region_positive"].fillna(False).astype(bool)
        summary_table["n_overlapping_optional_region_objects"] = (
            summary_table["n_overlapping_optional_region_objects"].fillna(0).astype(int)
        )
        summary_table["best_optional_region_overlap_voxels"] = (
            summary_table["best_optional_region_overlap_voxels"].fillna(0).astype(int)
        )
        summary_table["best_optional_region_overlap_fraction"] = (
            summary_table["best_optional_region_overlap_fraction"].fillna(0.0).astype(float)
        )
        summary_table["marker_and_optional_region_positive"] = (
            summary_table["marker_positive"] & summary_table["optional_region_positive"]
        )

    return summary_table


def _build_optional_region_summary_table(
    roi_labels_2d: np.ndarray,
    cell_masks: np.ndarray,
    optional_region_masks: np.ndarray | None,
    config: ColocalizationConfig,
) -> pd.DataFrame:
    """Summarize which cells overlap an optional third-channel segmentation.

    This produces one row per cell with overlap statistics against the
    segmented optional third channel so the main summary table can expose both
    separate third-channel positivity and marker-and-third-channel
    double-positivity.
    """

    if optional_region_masks is None:
        return pd.DataFrame(
            columns=[
                "roi_id",
                "cell_label",
                "optional_region_positive",
                "n_overlapping_optional_region_objects",
                "best_optional_region_label",
                "best_optional_region_overlap_voxels",
                "best_optional_region_overlap_fraction",
            ]
        )

    rows: list[dict[str, int | float | bool]] = []
    roi_ids = np.unique(roi_labels_2d)
    roi_ids = roi_ids[roi_ids != 0]

    for roi_id in roi_ids:
        roi_mask_2d = roi_labels_2d == roi_id
        bbox = get_bbox_2d(roi_mask_2d)
        if bbox is None:
            continue

        y_slice, x_slice = bbox
        cell_roi = cell_masks[:, y_slice, x_slice]
        optional_region_roi = optional_region_masks[:, y_slice, x_slice]
        detailed_rows = analyze_label_overlaps(cell_roi, optional_region_roi, roi_id=int(roi_id))
        detailed_table = pd.DataFrame(detailed_rows)
        if detailed_table.empty:
            continue

        for cell_label in np.unique(detailed_table["cell_label"]):
            detailed_cell = detailed_table[detailed_table["cell_label"] == cell_label]
            n_overlapping_objects = int(detailed_cell["n_overlapping_markers"].max())
            best_idx = detailed_cell["overlap_voxels"].idxmax()
            best_overlap_voxels = int(detailed_cell.loc[best_idx, "overlap_voxels"])
            best_overlap_fraction = float(detailed_cell.loc[best_idx, "overlap_fraction_of_cell"])
            best_optional_region_label = detailed_cell.loc[best_idx, "marker_label"]
            optional_region_positive = (
                (n_overlapping_objects > 0)
                and (best_overlap_voxels >= config.min_overlap_voxels)
                and (best_overlap_fraction >= config.overlap_fraction_threshold)
            )

            rows.append(
                {
                    "roi_id": int(roi_id),
                    "cell_label": int(cell_label),
                    "optional_region_positive": bool(optional_region_positive),
                    "n_overlapping_optional_region_objects": n_overlapping_objects,
                    "best_optional_region_label": (
                        int(best_optional_region_label)
                        if not pd.isna(best_optional_region_label)
                        else np.nan
                    ),
                    "best_optional_region_overlap_voxels": best_overlap_voxels,
                    "best_optional_region_overlap_fraction": best_overlap_fraction,
                }
            )

    return pd.DataFrame(rows)


def _build_overview_table(
    roi_labels_2d: np.ndarray,
    loaded_images: LoadedImageChannels,
    cell_masks: np.ndarray,
    marker_masks: np.ndarray,
    summary_table: pd.DataFrame,
    optional_region_masks: np.ndarray | None,
    analysis_z_bounds: tuple[int, int] | None,
) -> pd.DataFrame:
    """Create one ROI overview row per ROI.

    The overview combines ROI geometry, counts of segmented objects, counts of
    positive cells, and channel-wise occupancy metrics. When a global analysis
    z crop is active, ROI volume and all 3D occupancy metrics are computed only
    inside that z interval.
    """

    z_size_um, y_size_um, x_size_um = loaded_images.voxel_scale_zyx
    pixel_area_um2 = y_size_um * x_size_um
    voxel_volume_um3 = z_size_um * y_size_um * x_size_um
    n_z = loaded_images.cell_image.shape[0]
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
        roi_mask_3d = np.zeros((n_z, *roi_mask_2d.shape), dtype=bool)
        roi_mask_3d[z_start:z_stop] = np.repeat(
            roi_mask_2d[np.newaxis, :, :],
            analysis_depth,
            axis=0,
        )

        cell_labels_roi = np.unique(cell_masks[roi_mask_3d])
        cell_labels_roi = cell_labels_roi[cell_labels_roi != 0]

        marker_labels_roi = np.unique(marker_masks[roi_mask_3d])
        marker_labels_roi = marker_labels_roi[marker_labels_roi != 0]

        summary_roi = summary_table[summary_table["roi_id"] == roi_id]
        row: dict[str, int | float] = {
            "roi_id": int(roi_id),
            "n_cells": int(len(cell_labels_roi)),
            "n_marker_positive_cells": int(summary_roi["marker_positive"].sum()) if not summary_roi.empty else 0,
            "n_marker_objects": int(len(marker_labels_roi)),
            "drawn_roi_area_px": roi_area_px,
            "drawn_roi_area_um2": roi_area_um2,
            "roi_volume_voxels": roi_volume_voxels,
            "roi_volume_um3": roi_volume_um3,
        }

        if "optional_region_positive" in summary_roi.columns:
            row["n_optional_region_positive_cells"] = (
                int(summary_roi["optional_region_positive"].sum()) if not summary_roi.empty else 0
            )
        if "marker_and_optional_region_positive" in summary_roi.columns:
            row["n_marker_and_optional_region_positive_cells"] = (
                int(summary_roi["marker_and_optional_region_positive"].sum()) if not summary_roi.empty else 0
            )

        row.update(
            _compute_mask_occupancy_metrics(
                "cell",
                cell_masks,
                roi_mask_2d,
                loaded_images.voxel_scale_zyx,
                analysis_z_bounds,
            )
        )
        row.update(
            _compute_mask_occupancy_metrics(
                "marker",
                marker_masks,
                roi_mask_2d,
                loaded_images.voxel_scale_zyx,
                analysis_z_bounds,
            )
        )

        if optional_region_masks is not None:
            row.update(
                _compute_mask_occupancy_metrics(
                    "optional_region",
                    optional_region_masks,
                    roi_mask_2d,
                    loaded_images.voxel_scale_zyx,
                    analysis_z_bounds,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def _compute_mask_occupancy_metrics(
    prefix: str,
    label_image: np.ndarray,
    roi_mask_2d: np.ndarray,
    voxel_scale_zyx: tuple[float, float, float],
    analysis_z_bounds: tuple[int, int] | None,
) -> dict[str, int | float]:
    """Compute generic ROI occupancy metrics for one segmented channel.

    Both 2D projection coverage and true 3D occupancy are reported. When an
    ``analysis_z_bounds`` interval is provided, only voxels inside that z range
    contribute to the 3D denominator and numerator.
    """

    z_size_um, y_size_um, x_size_um = voxel_scale_zyx
    pixel_area_um2 = y_size_um * x_size_um
    voxel_volume_um3 = z_size_um * y_size_um * x_size_um
    n_z = label_image.shape[0]
    z_start, z_stop = analysis_z_bounds if analysis_z_bounds is not None else (0, n_z)
    analysis_depth = z_stop - z_start

    roi_area_px = int(roi_mask_2d.sum())
    roi_volume_voxels = int(roi_area_px * analysis_depth)
    roi_mask_3d = np.zeros((n_z, *roi_mask_2d.shape), dtype=bool)
    roi_mask_3d[z_start:z_stop] = np.repeat(roi_mask_2d[np.newaxis, :, :], analysis_depth, axis=0)
    occupancy_mask = (label_image > 0) & roi_mask_3d

    occupied_volume_voxels = int(occupancy_mask.sum())
    occupied_volume_um3 = float(occupied_volume_voxels * voxel_volume_um3)
    occupancy_3d_percent = float(100 * occupied_volume_voxels / roi_volume_voxels) if roi_volume_voxels > 0 else np.nan

    occupancy_projection_2d = occupancy_mask.any(axis=0)
    occupied_area_px = int((occupancy_projection_2d & roi_mask_2d).sum())
    occupied_area_um2 = float(occupied_area_px * pixel_area_um2)
    occupancy_2d_percent = float(100 * occupied_area_px / roi_area_px) if roi_area_px > 0 else np.nan

    return {
        f"{prefix}_occupancy_area_px_2d_projection": occupied_area_px,
        f"{prefix}_occupancy_area_um2_2d_projection": occupied_area_um2,
        f"{prefix}_occupancy_coverage_2d_percent": occupancy_2d_percent,
        f"{prefix}_occupancy_volume_voxels_3d": occupied_volume_voxels,
        f"{prefix}_occupancy_volume_um3_3d": occupied_volume_um3,
        f"{prefix}_occupancy_coverage_3d_percent": occupancy_3d_percent,
    }


def run_roi_cellpose_colocalization(
    loaded_images: LoadedImageChannels,
    roi_labels_2d: np.ndarray,
    cell_model_config: CellposeModelConfig,
    marker_model_config: CellposeModelConfig,
    colocalization_config: ColocalizationConfig,
    runtime_config: RuntimeConfig,
    optional_region_model_config: CellposeModelConfig | None = None,
    optional_region_result: OptionalRegionSegmentationResult | None = None,
) -> ColocalizationRunResult:
    """Run the configured ROI-wise segmentation workflow and build result tables.

    The pipeline always segments ROI crops in ``XY`` and may additionally apply
    one global analysis z crop resolved from the participating channel configs.
    That z crop affects all channels, all ROIs, and all downstream
    quantification consistently, while the exported and visualized arrays keep
    full-stack shape. When the input ``loaded_images`` bundle already
    represents a prepared z projection, segmentation and quantification operate
    on that projected 2D analysis view instead of the original full stack.

    The two primary analysis channels can each use either Cellpose or one of
    the supported threshold-based backends. An optional third channel can be
    segmented through the same mechanism and contributes occupancy metrics, and
    optionally per-cell positivity, to the result tables.
    """

    if not runtime_config.process_rois:
        raise ValueError("ROI processing is disabled in RuntimeConfig.")

    roi_ids = np.unique(roi_labels_2d)
    roi_ids = roi_ids[roi_ids != 0]

    print(f"Found {len(roi_ids)} ROIs: {roi_ids}")
    if loaded_images.z_projection_method is not None:
        analysis_z_bounds = None
    else:
        analysis_z_bounds = _resolve_analysis_z_bounds(
            loaded_images.cell_image.shape[0],
            cell_model_config,
            marker_model_config,
            optional_region_model_config,
        )
    z_slice = slice(*analysis_z_bounds) if analysis_z_bounds is not None else slice(None)

    cell_model, marker_model = create_cellpose_models_for_channels(
        cell_model_config=cell_model_config,
        marker_model_config=marker_model_config,
        use_gpu=runtime_config.use_gpu,
    )
    optional_region_model = None
    if (
        optional_region_model_config is not None
        and loaded_images.optional_region_image is not None
        and normalize_segmentation_method(optional_region_model_config.segmentation_method) == "cellpose"
    ):
        optional_region_model = create_cellpose_model(
            optional_region_model_config.model_name_or_path,
            runtime_config.use_gpu,
        )

    full_cell_masks = np.zeros(loaded_images.cell_image.shape, dtype=np.uint32)
    full_marker_masks = np.zeros(loaded_images.marker_image.shape, dtype=np.uint32)
    full_optional_region_masks = None
    if optional_region_model_config is not None and loaded_images.optional_region_image is not None:
        full_optional_region_masks = np.zeros(loaded_images.optional_region_image.shape, dtype=np.uint32)
    cell_roi_caches: list[CellposeRefinementRoiCache] = []
    marker_roi_caches: list[CellposeRefinementRoiCache] = []

    cell_label_offset = 0
    marker_label_offset = 0
    for roi_id in roi_ids:
        print(f"\nProcessing ROI {int(roi_id)}...")
        roi_mask_2d = roi_labels_2d == roi_id
        bbox = get_bbox_2d(roi_mask_2d)
        if bbox is None:
            print(f"Skipping ROI {int(roi_id)}: empty ROI")
            continue

        y_slice, x_slice = bbox
        roi_mask_crop_2d = roi_mask_2d[y_slice, x_slice]

        cell_crop = loaded_images.cell_image[z_slice, y_slice, x_slice].copy()
        marker_crop = loaded_images.marker_image[z_slice, y_slice, x_slice].copy()
        cell_crop = apply_prefilter(cell_crop, cell_model_config)
        marker_crop = apply_prefilter(marker_crop, marker_model_config)
        cell_crop[:, ~roi_mask_crop_2d] = 0
        marker_crop[:, ~roi_mask_crop_2d] = 0
        optional_region_masks_roi = None
        if optional_region_model_config is not None:
            if loaded_images.optional_region_image is None:
                raise ValueError(
                    "An optional-region segmentation config was provided, but "
                    "no optional region channel was loaded."
                )
            optional_region_crop = loaded_images.optional_region_image[z_slice, y_slice, x_slice].copy()
            optional_region_crop = apply_prefilter(optional_region_crop, optional_region_model_config)
            optional_region_crop[:, ~roi_mask_crop_2d] = 0

        cell_masks_roi, cell_refinement_cache = evaluate_segmentation_method(
            cell_model,
            cell_crop,
            cell_model_config,
            loaded_images.voxel_scale_zyx,
        )
        marker_masks_roi, marker_refinement_cache = evaluate_segmentation_method(
            marker_model,
            marker_crop,
            marker_model_config,
            loaded_images.voxel_scale_zyx,
        )
        if optional_region_model_config is not None:
            optional_region_masks_roi, _ = evaluate_segmentation_method(
                optional_region_model,
                optional_region_crop,
                optional_region_model_config,
                loaded_images.voxel_scale_zyx,
            )

        if cell_refinement_cache is not None:
            cell_refinement_cache.roi_id = int(roi_id)
            cell_refinement_cache.y_min = int(y_slice.start)
            cell_refinement_cache.y_max = int(y_slice.stop)
            cell_refinement_cache.x_min = int(x_slice.start)
            cell_refinement_cache.x_max = int(x_slice.stop)
            cell_refinement_cache.roi_mask_crop_2d = roi_mask_crop_2d.copy()
            cell_roi_caches.append(cell_refinement_cache)
        if marker_refinement_cache is not None:
            marker_refinement_cache.roi_id = int(roi_id)
            marker_refinement_cache.y_min = int(y_slice.start)
            marker_refinement_cache.y_max = int(y_slice.stop)
            marker_refinement_cache.x_min = int(x_slice.start)
            marker_refinement_cache.x_max = int(x_slice.stop)
            marker_refinement_cache.roi_mask_crop_2d = roi_mask_crop_2d.copy()
            marker_roi_caches.append(marker_refinement_cache)

        cell_masks_roi = relabel_with_offset(cell_masks_roi, cell_label_offset)
        marker_masks_roi = relabel_with_offset(marker_masks_roi, marker_label_offset)

        if cell_masks_roi.max() > 0:
            cell_label_offset = int(cell_masks_roi.max())
        if marker_masks_roi.max() > 0:
            marker_label_offset = int(marker_masks_roi.max())

        full_cell_masks[z_slice, y_slice, x_slice] = np.maximum(
            full_cell_masks[z_slice, y_slice, x_slice],
            cell_masks_roi,
        )
        full_marker_masks[z_slice, y_slice, x_slice] = np.maximum(
            full_marker_masks[z_slice, y_slice, x_slice],
            marker_masks_roi,
        )
        if full_optional_region_masks is not None and optional_region_masks_roi is not None:
            full_optional_region_masks[z_slice, y_slice, x_slice] = np.maximum(
                full_optional_region_masks[z_slice, y_slice, x_slice],
                optional_region_masks_roi,
            )

    cell_refinement_context = None
    marker_refinement_context = None
    if cell_roi_caches:
        cell_refinement_context = CellposeChannelRefinementContext(
            model=cell_model,
            model_name_or_path=cell_model_config.model_name_or_path,
            roi_caches=cell_roi_caches,
        )
    if marker_roi_caches:
        marker_refinement_context = CellposeChannelRefinementContext(
            model=marker_model,
            model_name_or_path=marker_model_config.model_name_or_path,
            roi_caches=marker_roi_caches,
        )

    return analyze_existing_masks(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        cell_masks=full_cell_masks,
        marker_masks=full_marker_masks,
        colocalization_config=colocalization_config,
        optional_region_result=optional_region_result,
        optional_region_masks=full_optional_region_masks,
        analysis_z_bounds=analysis_z_bounds,
        cell_refinement_context=cell_refinement_context,
        marker_refinement_context=marker_refinement_context,
        cell_model_config=cell_model_config,
        marker_model_config=marker_model_config,
    )
# %% END
