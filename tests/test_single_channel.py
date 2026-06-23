from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellcoloc import CellposeModelConfig, RuntimeConfig, SingleChannelAnalysisConfig
from cellcoloc.io import build_results_paths, load_analysis_images
from cellcoloc.schemas import (
    CellposeChannelRefinementContext,
    CellposeRefinementRoiCache,
    LoadedSingleChannelImage,
    SingleChannelRunResult,
    SingleChannelTables,
)
from cellcoloc.single_channel import (
    _build_single_channel_object_table,
    _build_single_channel_overview_table,
    _build_single_channel_plausibility_table,
    _compute_3d_ellipticity,
    _compute_3d_surface_area_um2,
    analyze_existing_single_channel_masks,
    build_single_channel_results_paths,
    extract_single_channel_masks_from_viewer,
    export_single_channel_outputs,
    load_single_channel_image,
    prepare_loaded_single_channel_image_for_analysis,
    refine_single_channel_run_result_from_cellpose_cache,
    run_roi_single_channel_segmentation,
    show_single_channel_results,
    try_load_single_channel_roi_labels,
)


class _FakeLayer:
    def __init__(self, data=None, **kwargs):
        self.data = data
        self.visible = True
        self.scale = kwargs.get("scale")
        self.blending = kwargs.get("blending")
        self.colormap = kwargs.get("colormap")
        self.size = kwargs.get("size")
        self.face_color = kwargs.get("face_color")
        self.text = kwargs.get("text")


class _FakeLayerList(dict):
    pass


class _FakeDims:
    ndim = 3


class _FakeViewer:
    _instances: list["_FakeViewer"] = []

    def __init__(self):
        self.layers = _FakeLayerList()
        self.dims = _FakeDims()
        type(self)._instances.append(self)

    def add_image(self, data, name, **kwargs):
        self.layers[name] = _FakeLayer(data, **kwargs)

    def add_labels(self, data, name, **kwargs):
        self.layers[name] = _FakeLayer(data, **kwargs)

    def add_points(self, data, name, **kwargs):
        self.layers[name] = _FakeLayer(data, **kwargs)


class _FakeRefinementModel:
    def _compute_masks(self, shape_for_masks, dP, cellprob, **kwargs):
        if kwargs.get("do_3D"):
            return np.ones(shape_for_masks, dtype=np.uint32)
        return np.ones((shape_for_masks[1], shape_for_masks[2]), dtype=np.uint32)


def _make_loaded_single_channel(tmp_path, *, z=1) -> LoadedSingleChannelImage:
    source = tmp_path / "single.ome.tif"
    source.write_bytes(b"fake")
    image = np.zeros((z, 6, 6), dtype=np.float32)
    image[:, 1:3, 1:3] = 10
    image[:, 3:5, 3:5] = 8
    return LoadedSingleChannelImage(
        source_path=source,
        paths=build_single_channel_results_paths(source),
        voxel_scale_zyx=(2.0, 0.5, 0.5),
        image=image,
        raw_shape_tzcyx=(1, z, 1, 6, 6),
        raw_z_size=z,
        is_3d=z > 1,
        metadata={},
    )


def test_single_channel_geometry_helpers() -> None:
    mask = np.zeros((2, 3, 3), dtype=bool)
    mask[:, 1, 1] = True
    assert _compute_3d_surface_area_um2(mask, (2.0, 0.5, 0.5)) > 0
    ell = _compute_3d_ellipticity(mask, (2.0, 0.5, 0.5))
    assert np.isnan(ell) or 0.0 <= ell <= 1.0


def test_single_channel_table_builders_2d_and_3d(tmp_path) -> None:
    roi_labels = np.ones((6, 6), dtype=np.uint16)

    loaded_2d = _make_loaded_single_channel(tmp_path, z=1)
    masks_2d = np.zeros((1, 6, 6), dtype=np.uint32)
    masks_2d[0, 1:3, 1:3] = 1
    object_table_2d = _build_single_channel_object_table(masks_2d, loaded_2d, roi_labels)
    assert "object_roundness_2d" in object_table_2d.columns
    assert pd.notna(object_table_2d.loc[0, "object_area_px_2d"])

    loaded_3d = _make_loaded_single_channel(tmp_path, z=2)
    masks_3d = np.zeros((2, 6, 6), dtype=np.uint32)
    masks_3d[:, 1:3, 1:3] = 1
    object_table_3d = _build_single_channel_object_table(masks_3d, loaded_3d, roi_labels)
    assert "object_sphericity_3d" in object_table_3d.columns
    assert pd.notna(object_table_3d.loc[0, "object_volume_voxels_3d"])

    plausibility = _build_single_channel_plausibility_table(masks_3d, roi_labels)
    assert "object_voxels - object_voxels_props" in plausibility.columns

    overview = _build_single_channel_overview_table(
        roi_labels_2d=roi_labels,
        loaded_image=loaded_3d,
        masks=masks_3d,
        object_table=object_table_3d,
        analysis_z_bounds=None,
    )
    assert overview.iloc[0]["n_objects"] == 1
    assert "average_object_volume_um3_3d" in overview.columns


def test_single_channel_threshold_pipeline_and_refinement_passthrough(tmp_path) -> None:
    loaded = _make_loaded_single_channel(tmp_path, z=1)
    roi_labels = np.ones((6, 6), dtype=np.uint16)
    model_cfg = CellposeModelConfig(
        model_name_or_path="cpsam",
        segmentation_method="otsu",
        threshold_min_object_voxels=1,
        threshold_min_hole_voxels=0,
    )
    analysis_cfg = SingleChannelAnalysisConfig(min_object_voxels=1)
    runtime_cfg = RuntimeConfig(process_rois=True, use_gpu=False, open_results=False, draw_rois=False)

    run_result = run_roi_single_channel_segmentation(
        loaded_image=loaded,
        roi_labels_2d=roi_labels,
        model_config=model_cfg,
        analysis_config=analysis_cfg,
        runtime_config=runtime_cfg,
    )
    assert not run_result.tables.objects.empty

    refined = refine_single_channel_run_result_from_cellpose_cache(
        loaded_image=loaded,
        roi_labels_2d=roi_labels,
        run_result=run_result,
        analysis_config=analysis_cfg,
        model_config=None,
    )
    assert refined.tables.overview.iloc[0]["n_objects"] >= 1


def test_single_channel_projection_and_export(tmp_path) -> None:
    loaded = _make_loaded_single_channel(tmp_path, z=3)
    cfg = CellposeModelConfig(model_name_or_path="cpsam", z_crop=(1, 3), z_projection="mean")
    prepared = prepare_loaded_single_channel_image_for_analysis(loaded, cfg)
    assert prepared.image.shape == (1, 6, 6)
    assert prepared.z_projection_method == "mean"

    tables = SingleChannelTables(
        objects=pd.DataFrame([{"roi_id": 1, "object_label": 1}]),
        voxel_plausibility=pd.DataFrame([{"roi_id": 1, "object_label": 1}]),
        overview=pd.DataFrame([{"roi_id": 1, "n_objects": 1}]),
    )
    run_result = SingleChannelRunResult(
        masks=np.ones((1, 6, 6), dtype=np.uint32),
        tables=tables,
    )
    export_single_channel_outputs(run_result, loaded.paths)
    assert loaded.paths.object_csv_path.exists()
    assert loaded.paths.excel_path.exists()
    assert loaded.paths.mask_path.exists()


def test_single_channel_postfilter_and_missing_cache_error(tmp_path) -> None:
    loaded = _make_loaded_single_channel(tmp_path, z=1)
    roi_labels = np.ones((6, 6), dtype=np.uint16)
    masks = np.zeros((1, 6, 6), dtype=np.uint32)
    masks[0, 1:3, 1:3] = 1
    loaded.image[0, 1:3, 1:3] = 100
    cfg = CellposeModelConfig(
        model_name_or_path="cpsam",
        postfilters="min_intensity",
        min_intensity_measure="max",
        min_intensity_threshold=50,
    )
    result = analyze_existing_single_channel_masks(
        loaded_image=loaded,
        roi_labels_2d=roi_labels,
        masks=masks,
        analysis_config=SingleChannelAnalysisConfig(min_object_voxels=1),
        model_config=cfg,
    )
    assert result.tables.overview.iloc[0]["n_objects"] == 1

    with pytest.raises(ValueError):
        refine_single_channel_run_result_from_cellpose_cache(
            loaded_image=loaded,
            roi_labels_2d=roi_labels,
            run_result=result,
            analysis_config=SingleChannelAnalysisConfig(min_object_voxels=1),
            model_config=CellposeModelConfig(model_name_or_path="cpsam"),
        )


def test_single_channel_loaders_and_viewer_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    image = np.zeros((1, 1, 2, 5, 5), dtype=np.uint16)
    image[0, 0, 1, 2, 2] = 10
    metadata = {
        "PhysicalSizeZ": 1.0,
        "PhysicalSizeY": 0.5,
        "PhysicalSizeX": 0.5,
        "PhysicalSizeZUnit": "micron",
        "PhysicalSizeYUnit": "micron",
        "PhysicalSizeXUnit": "micron",
    }

    def fake_imread(path, zarr_store=None, reuse_disk_cache=False):
        return image, metadata

    monkeypatch.setattr("cellcoloc.single_channel.om.imread", fake_imread)
    source = tmp_path / "single.ome.tif"
    source.write_bytes(b"fake")

    loaded = load_single_channel_image(
        source_path=source,
        channel_config=__import__("cellcoloc").SingleChannelConfig(channel_index=1),
        voxel_scale_zyx=None,
    )
    assert loaded.image.shape == (1, 5, 5)
    assert try_load_single_channel_roi_labels(tmp_path / "missing.tif") is None

    viewer = _FakeViewer()
    run_result = SingleChannelRunResult(
        masks=np.ones((1, 5, 5), dtype=np.uint32),
        tables=SingleChannelTables(
            objects=pd.DataFrame([{"roi_id": 1, "object_label": 1}]),
            voxel_plausibility=pd.DataFrame([{"roi_id": 1, "object_label": 1}]),
            overview=pd.DataFrame([{"roi_id": 1, "n_objects": 1}]),
        ),
    )
    viewer = show_single_channel_results(
        loaded_image=loaded,
        roi_labels_2d=np.ones((5, 5), dtype=np.uint16),
        run_result=run_result,
        display_names=__import__("cellcoloc").SingleChannelDisplayNames(channel="DAPI", objects="Objs"),
        viewer=viewer,
        layers_to_show=["channel_image", "rois", "roi_numbers", "masks"],
    )
    assert "DAPI" in viewer.layers
    assert "ROIs" in viewer.layers
    assert "ROI numbers" in viewer.layers
    assert "Objs" in viewer.layers
    masks = extract_single_channel_masks_from_viewer(viewer, object_layer_name="Objs")
    assert masks.shape == (1, 5, 5)


def test_single_channel_refinement_with_fake_cache(tmp_path) -> None:
    loaded = _make_loaded_single_channel(tmp_path, z=1)
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
    context = CellposeChannelRefinementContext(
        model=_FakeRefinementModel(),
        model_name_or_path="cpsam",
        roi_caches=[roi_cache],
    )
    result = SingleChannelRunResult(
        masks=np.zeros((1, 6, 6), dtype=np.uint32),
        tables=SingleChannelTables(
            objects=pd.DataFrame(),
            voxel_plausibility=pd.DataFrame(),
            overview=pd.DataFrame(),
        ),
        refinement_context=context,
    )
    refined = refine_single_channel_run_result_from_cellpose_cache(
        loaded_image=loaded,
        roi_labels_2d=roi_labels,
        run_result=result,
        analysis_config=SingleChannelAnalysisConfig(min_object_voxels=1),
        model_config=CellposeModelConfig(model_name_or_path="cpsam"),
    )
    assert refined.masks.max() >= 1
