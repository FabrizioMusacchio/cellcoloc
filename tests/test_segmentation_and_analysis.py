from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellcoloc import (
    CellposeModelConfig,
    ColocalizationConfig,
    OptionalRegionSegmentationConfig,
    RuntimeConfig,
)
from cellcoloc.analysis import (
    _apply_analysis_z_bounds,
    _compute_mask_occupancy_metrics,
    _normalize_z_crop_bounds,
    _normalize_z_projection_method,
    _project_zyx_volume,
    _resolve_analysis_z_bounds,
    analyze_existing_masks,
    analyze_label_overlaps,
    build_positive_cell_mask,
    prepare_loaded_images_for_analysis,
    refine_run_result_from_cellpose_cache,
    run_roi_cellpose_colocalization,
)
from cellcoloc.io import build_results_paths
from cellcoloc.schemas import CellposeChannelRefinementContext, CellposeRefinementRoiCache, LoadedImageChannels
from cellcoloc.segmentation import (
    _legacy_threshold_to_max_size,
    filter_labels_by_size,
    normalize_segmentation_method,
    relabel_with_offset,
    resolve_cellpose_anisotropy,
    segment_optional_region,
    segment_threshold_channel,
)


def _make_loaded_images(tmp_path, *, z=1) -> LoadedImageChannels:
    source = tmp_path / "sample.ome.tif"
    source.write_bytes(b"fake")
    shape = (z, 6, 6)
    cell = np.zeros(shape, dtype=np.float32)
    marker = np.zeros(shape, dtype=np.float32)
    optional = np.zeros(shape, dtype=np.float32)
    cell[:, 1:3, 1:3] = 10
    cell[:, 3:5, 3:5] = 8
    marker[:, 1:3, 1:3] = 10
    marker[:, 4:5, 4:5] = 10
    optional[:, 3:5, 3:5] = 10
    return LoadedImageChannels(
        source_path=source,
        paths=build_results_paths(source),
        voxel_scale_zyx=(2.0, 0.5, 0.5),
        cell_image=cell,
        marker_image=marker,
        optional_region_image=optional,
        raw_shape_tzcyx=(1, z, 3, 6, 6),
        raw_z_size=z,
        is_3d=z > 1,
        metadata={},
    )


class _FakeRefinementModel:
    def _compute_masks(self, shape_for_masks, dP, cellprob, **kwargs):
        if kwargs.get("do_3D"):
            return np.ones(shape_for_masks, dtype=np.uint32)
        return np.ones((shape_for_masks[1], shape_for_masks[2]), dtype=np.uint32)


def test_segmentation_helpers_and_anisotropy_logic() -> None:
    assert normalize_segmentation_method("OTSU") == "otsu"
    assert _legacy_threshold_to_max_size(5) == 4
    assert np.array_equal(relabel_with_offset(np.array([[0, 1]]), 2), np.array([[0, 3]]))
    assert np.array_equal(
        filter_labels_by_size(np.array([[0, 1, 1, 2]], dtype=np.uint32), 2),
        np.array([[0, 1, 1, 0]], dtype=np.uint32),
    )

    cfg = CellposeModelConfig(model_name_or_path="cpsam", anisotropy=True)
    assert resolve_cellpose_anisotropy(cfg, (3.0, 1.0, 1.0), do_3d=True) == pytest.approx(3.0)
    assert resolve_cellpose_anisotropy(cfg, (1.0, 1.0, 1.0), do_3d=True) is None


def test_segment_threshold_channel_and_optional_region() -> None:
    image = np.zeros((1, 6, 6), dtype=np.float32)
    image[0, 1:3, 1:3] = 10
    image[0, 4:6, 4:6] = 5
    cfg = CellposeModelConfig(
        model_name_or_path="cpsam",
        segmentation_method="otsu",
        threshold_min_object_voxels=1,
        threshold_min_hole_voxels=0,
    )

    labels = segment_threshold_channel(image, cfg)
    assert labels.shape == image.shape
    assert labels.max() >= 1

    roi = np.ones((6, 6), dtype=np.uint16)
    region = segment_optional_region(
        image,
        roi,
        config=OptionalRegionSegmentationConfig(
            enabled=True,
            method="otsu",
            gaussian_sigma=None,
            background_sigma=None,
            min_object_voxels=1,
            min_hole_voxels=0,
            apply_closing=False,
        ),
    )
    assert region.labels.shape == image.shape
    assert region.threshold > 0


def test_projection_and_z_bounds_helpers(tmp_path) -> None:
    loaded = _make_loaded_images(tmp_path, z=3)
    cfg = CellposeModelConfig(model_name_or_path="cpsam", z_crop=(1, 3), z_projection="max")

    assert _normalize_z_crop_bounds((1, None), 4) == (1, 4)
    assert _normalize_z_projection_method("Mean") == "mean"
    assert _resolve_analysis_z_bounds(3, cfg) == (1, 3)
    projected = _project_zyx_volume(loaded.cell_image, "max")
    assert projected.shape == (1, 6, 6)

    prepared = prepare_loaded_images_for_analysis(loaded, cfg)
    assert prepared.cell_image.shape == (1, 6, 6)
    assert prepared.z_projection_method == "max"
    assert prepared.analysis_z_bounds == (1, 3)


def test_multichannel_threshold_pipeline_end_to_end(tmp_path) -> None:
    loaded = _make_loaded_images(tmp_path, z=1)
    roi_labels = np.ones((6, 6), dtype=np.uint16)
    cell_cfg = CellposeModelConfig(
        model_name_or_path="cpsam",
        segmentation_method="otsu",
        threshold_min_object_voxels=1,
        threshold_min_hole_voxels=0,
    )
    marker_cfg = CellposeModelConfig(
        model_name_or_path="cpsam",
        segmentation_method="otsu",
        threshold_min_object_voxels=1,
        threshold_min_hole_voxels=0,
    )
    optional_cfg = CellposeModelConfig(
        model_name_or_path="cpsam",
        segmentation_method="otsu",
        threshold_min_object_voxels=1,
        threshold_min_hole_voxels=0,
    )
    coloc_cfg = ColocalizationConfig(
        min_cell_voxels=1,
        min_overlap_voxels=1,
        overlap_fraction_threshold=0.2,
        evaluate_optional_region_cell_positivity=True,
    )
    runtime_cfg = RuntimeConfig(process_rois=True, use_gpu=False, open_results=False, draw_rois=False)

    run_result = run_roi_cellpose_colocalization(
        loaded_images=loaded,
        roi_labels_2d=roi_labels,
        cell_model_config=cell_cfg,
        marker_model_config=marker_cfg,
        optional_region_model_config=optional_cfg,
        colocalization_config=coloc_cfg,
        runtime_config=runtime_cfg,
    )

    assert not run_result.tables.summary.empty
    assert "optional_region_positive" in run_result.tables.summary.columns
    assert "cell_area_px_2d" in run_result.tables.summary.columns
    assert "cell_roundness_2d" in run_result.tables.summary.columns
    assert "n_optional_region_positive_cells" in run_result.tables.overview.columns
    assert run_result.tables.marker_properties is not None
    assert "marker_area_px_2d" in run_result.tables.marker_properties.columns
    assert run_result.tables.third_channel_properties is not None
    assert "optional_region_area_px_2d" in run_result.tables.third_channel_properties.columns
    assert run_result.tables.roi_cell_summary is not None
    assert "average_cell_area_px_2d" in run_result.tables.roi_cell_summary.columns
    assert run_result.tables.roi_marker_summary is not None
    assert "average_marker_area_px_2d" in run_result.tables.roi_marker_summary.columns
    assert run_result.tables.roi_third_channel_summary is not None
    assert "average_optional_region_area_px_2d" in run_result.tables.roi_third_channel_summary.columns

    refined = refine_run_result_from_cellpose_cache(
        loaded_images=loaded,
        roi_labels_2d=roi_labels,
        run_result=run_result,
        colocalization_config=coloc_cfg,
        cell_model_config=None,
        marker_model_config=None,
        optional_region_model_config=None,
    )
    assert refined.tables.overview.iloc[0]["n_cells"] >= 1


def test_analysis_helpers_on_existing_masks(tmp_path) -> None:
    loaded = _make_loaded_images(tmp_path, z=2)
    roi_labels = np.ones((6, 6), dtype=np.uint16)
    cell_masks = np.zeros((2, 6, 6), dtype=np.uint32)
    marker_masks = np.zeros((2, 6, 6), dtype=np.uint32)
    cell_masks[:, 1:3, 1:3] = 1
    cell_masks[:, 3:5, 3:5] = 2
    marker_masks[:, 1:3, 1:3] = 1

    rows = analyze_label_overlaps(cell_masks, marker_masks, roi_id=1)
    detailed = pd.DataFrame(rows)
    assert set(detailed["cell_label"]) == {1, 2}

    summary_input = pd.DataFrame(
        [{"cell_label": 1, "marker_positive": True}, {"cell_label": 2, "marker_positive": False}]
    )
    positive = build_positive_cell_mask(cell_masks, summary_input)
    assert set(np.unique(positive)) == {0, 1}

    analyzed = analyze_existing_masks(
        loaded_images=loaded,
        roi_labels_2d=roi_labels,
        cell_masks=cell_masks,
        marker_masks=marker_masks,
        colocalization_config=ColocalizationConfig(min_cell_voxels=1, min_overlap_voxels=1, overlap_fraction_threshold=0.2),
        analysis_z_bounds=(0, 1),
    )
    assert analyzed.analysis_z_bounds == (0, 1)
    assert analyzed.tables.overview.iloc[0]["cell_occupancy_volume_voxels_3d"] > 0
    assert "cell_volume_voxels_3d" in analyzed.tables.summary.columns
    assert analyzed.tables.marker_properties is not None
    assert "marker_volume_voxels_3d" in analyzed.tables.marker_properties.columns

    cropped = _apply_analysis_z_bounds(cell_masks, (0, 1))
    assert np.count_nonzero(cropped[1]) == 0
    metrics = _compute_mask_occupancy_metrics("cell", cell_masks, roi_labels.astype(bool), loaded.voxel_scale_zyx, (0, 1))
    assert metrics["cell_occupancy_area_px_2d_projection"] > 0


def test_refine_run_result_from_cellpose_cache_uses_fake_contexts(tmp_path) -> None:
    loaded = _make_loaded_images(tmp_path, z=1)
    roi_labels = np.ones((6, 6), dtype=np.uint16)
    roi_cache = CellposeRefinementRoiCache(
        roi_id=1,
        y_min=0,
        y_max=6,
        x_min=0,
        x_max=6,
        roi_mask_crop_2d=np.ones((6, 6), dtype=bool),
        shape_for_masks=(1, 6, 6),
        dP=np.ones((2, 1, 6, 6), dtype=np.float32),
        cellprob=np.ones((1, 6, 6), dtype=np.float32),
        do_3d=False,
        niter=5,
        min_size=1,
        max_size_fraction=0.4,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
    )
    refinement_context = CellposeChannelRefinementContext(
        model=_FakeRefinementModel(),
        model_name_or_path="cpsam",
        roi_caches=[roi_cache],
    )
    empty_tables = pd.DataFrame()
    run_result = __import__("cellcoloc").ColocalizationRunResult(
        cell_masks=np.zeros((1, 6, 6), dtype=np.uint32),
        marker_masks=np.zeros((1, 6, 6), dtype=np.uint32),
        positive_cell_masks=np.zeros((1, 6, 6), dtype=np.uint32),
        optional_region_masks=np.zeros((1, 6, 6), dtype=np.uint32),
        tables=__import__("cellcoloc").ColocalizationTables(
            detailed=empty_tables,
            summary=empty_tables,
            overview=empty_tables,
        ),
        cell_refinement_context=refinement_context,
        marker_refinement_context=refinement_context,
        optional_region_refinement_context=refinement_context,
    )

    refined = refine_run_result_from_cellpose_cache(
        loaded_images=loaded,
        roi_labels_2d=roi_labels,
        run_result=run_result,
        colocalization_config=ColocalizationConfig(min_cell_voxels=1, min_overlap_voxels=1, overlap_fraction_threshold=0.1),
        cell_model_config=CellposeModelConfig(model_name_or_path="cpsam"),
        marker_model_config=CellposeModelConfig(model_name_or_path="cpsam"),
        optional_region_model_config=CellposeModelConfig(model_name_or_path="cpsam"),
    )
    assert refined.cell_masks.max() >= 1
    assert refined.marker_masks.max() >= 1
