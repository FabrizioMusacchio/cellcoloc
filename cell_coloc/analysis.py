"""Core Cellpose colocalization analysis logic."""

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
    create_cellpose_models_for_channels,
    evaluate_segmentation_method,
    filter_labels_by_size,
    relabel_with_offset,
)


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
    """Create a label image containing only marker-positive cells."""

    if summary_table.empty:
        return np.zeros_like(cell_masks, dtype=np.uint32)

    positive_labels = summary_table.loc[summary_table["marker_positive"], "cell_label"].astype(np.uint32).to_numpy()
    max_label = int(cell_masks.max())
    positive_labels = positive_labels[positive_labels <= max_label]

    lookup = np.zeros(max_label + 1, dtype=np.uint32)
    lookup[positive_labels] = positive_labels
    return lookup[cell_masks]


def analyze_existing_masks(
    loaded_images: LoadedImageChannels,
    roi_labels_2d: np.ndarray,
    cell_masks: np.ndarray,
    marker_masks: np.ndarray,
    colocalization_config: ColocalizationConfig,
    optional_region_result: OptionalRegionSegmentationResult | None = None,
    cell_refinement_context: CellposeChannelRefinementContext | None = None,
    marker_refinement_context: CellposeChannelRefinementContext | None = None,
    cell_model_config: CellposeModelConfig | None = None,
    marker_model_config: CellposeModelConfig | None = None,
) -> ColocalizationRunResult:
    """Recompute colocalization tables from existing label masks.

    This helper is used both after the initial Cellpose segmentation and after
    any later manual or threshold-based refinement of the label masks.
    """

    full_cell_masks = np.asarray(cell_masks, dtype=np.uint32).copy()
    full_marker_masks = np.asarray(marker_masks, dtype=np.uint32).copy()

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

    summary_table = _build_summary_table(detailed_table, full_cell_masks, colocalization_config)
    overview_table = _build_overview_table(
        roi_labels_2d=roi_labels_2d,
        loaded_images=loaded_images,
        cell_masks=full_cell_masks,
        marker_masks=full_marker_masks,
        summary_table=summary_table,
        optional_region_result=optional_region_result,
    )
    positive_cell_masks = build_positive_cell_mask(full_cell_masks, summary_table)

    return ColocalizationRunResult(
        cell_masks=full_cell_masks,
        marker_masks=full_marker_masks,
        positive_cell_masks=positive_cell_masks,
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
    """Recompute full-size masks from cached Cellpose network outputs."""

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
    """

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

    return analyze_existing_masks(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        cell_masks=rebuilt_cell_masks,
        marker_masks=rebuilt_marker_masks,
        colocalization_config=colocalization_config,
        optional_region_result=optional_region_result,
        cell_refinement_context=run_result.cell_refinement_context,
        marker_refinement_context=run_result.marker_refinement_context,
        cell_model_config=cell_model_config,
        marker_model_config=marker_model_config,
    )


def _build_summary_table(
    detailed_table: pd.DataFrame,
    cell_masks: np.ndarray,
    config: ColocalizationConfig,
) -> pd.DataFrame:
    """Aggregate detailed overlap rows into one summary row per cell."""

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
    return summary_table


def _build_overview_table(
    roi_labels_2d: np.ndarray,
    loaded_images: LoadedImageChannels,
    cell_masks: np.ndarray,
    marker_masks: np.ndarray,
    summary_table: pd.DataFrame,
    optional_region_result: OptionalRegionSegmentationResult | None,
) -> pd.DataFrame:
    """Create one ROI overview row per ROI."""

    z_size_um, y_size_um, x_size_um = loaded_images.voxel_scale_zyx
    pixel_area_um2 = y_size_um * x_size_um
    voxel_volume_um3 = z_size_um * y_size_um * x_size_um
    n_z = loaded_images.cell_image.shape[0]

    rows: list[dict[str, int | float]] = []
    for roi_id in np.unique(roi_labels_2d):
        if roi_id == 0:
            continue

        roi_mask_2d = roi_labels_2d == roi_id
        roi_area_px = int(roi_mask_2d.sum())
        roi_area_um2 = float(roi_area_px * pixel_area_um2)
        roi_volume_voxels = int(roi_area_px * n_z)
        roi_volume_um3 = float(roi_volume_voxels * voxel_volume_um3)
        roi_mask_3d = np.repeat(roi_mask_2d[np.newaxis, :, :], n_z, axis=0)

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

        if optional_region_result is not None:
            region_mask_roi = optional_region_result.mask & roi_mask_3d
            region_volume_voxels = int(region_mask_roi.sum())
            region_volume_um3 = float(region_volume_voxels * voxel_volume_um3)
            region_coverage_3d_percent = float(100 * region_volume_voxels / roi_volume_voxels) if roi_volume_voxels > 0 else np.nan

            region_projection_2d = optional_region_result.mask.any(axis=0)
            region_area_px = int((region_projection_2d & roi_mask_2d).sum())
            region_area_um2 = float(region_area_px * pixel_area_um2)
            region_coverage_2d_percent = float(100 * region_area_px / roi_area_px) if roi_area_px > 0 else np.nan

            row.update(
                {
                    "optional_region_area_px_2d_projection": region_area_px,
                    "optional_region_area_um2_2d_projection": region_area_um2,
                    "optional_region_coverage_2d_percent": region_coverage_2d_percent,
                    "optional_region_volume_voxels_3d": region_volume_voxels,
                    "optional_region_volume_um3_3d": region_volume_um3,
                    "optional_region_coverage_3d_percent": region_coverage_3d_percent,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


def run_roi_cellpose_colocalization(
    loaded_images: LoadedImageChannels,
    roi_labels_2d: np.ndarray,
    cell_model_config: CellposeModelConfig,
    marker_model_config: CellposeModelConfig,
    colocalization_config: ColocalizationConfig,
    runtime_config: RuntimeConfig,
    optional_region_result: OptionalRegionSegmentationResult | None = None,
) -> ColocalizationRunResult:
    """Run the configured ROI-wise segmentation workflow and build result tables."""

    if not runtime_config.process_rois:
        raise ValueError("ROI processing is disabled in RuntimeConfig.")

    roi_ids = np.unique(roi_labels_2d)
    roi_ids = roi_ids[roi_ids != 0]

    print(f"Found {len(roi_ids)} ROIs: {roi_ids}")

    cell_model, marker_model = create_cellpose_models_for_channels(
        cell_model_config=cell_model_config,
        marker_model_config=marker_model_config,
        use_gpu=runtime_config.use_gpu,
    )

    full_cell_masks = np.zeros(loaded_images.cell_image.shape, dtype=np.uint32)
    full_marker_masks = np.zeros(loaded_images.marker_image.shape, dtype=np.uint32)
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

        cell_crop = loaded_images.cell_image[:, y_slice, x_slice].copy()
        marker_crop = loaded_images.marker_image[:, y_slice, x_slice].copy()
        cell_crop = apply_prefilter(cell_crop, cell_model_config)
        marker_crop = apply_prefilter(marker_crop, marker_model_config)
        cell_crop[:, ~roi_mask_crop_2d] = 0
        marker_crop[:, ~roi_mask_crop_2d] = 0

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

        full_cell_masks[:, y_slice, x_slice] = np.maximum(full_cell_masks[:, y_slice, x_slice], cell_masks_roi)
        full_marker_masks[:, y_slice, x_slice] = np.maximum(full_marker_masks[:, y_slice, x_slice], marker_masks_roi)

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
        cell_refinement_context=cell_refinement_context,
        marker_refinement_context=marker_refinement_context,
        cell_model_config=cell_model_config,
        marker_model_config=marker_model_config,
    )
