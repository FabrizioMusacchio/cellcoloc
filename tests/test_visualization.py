from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellcoloc import ColocalizationRunResult, DisplayNames
from cellcoloc.io import build_results_paths
from cellcoloc.schemas import (
    ColocalizationTables,
    LoadedImageChannels,
    OptionalRegionSegmentationResult,
)
from cellcoloc.visualization import (
    _build_roi_labels_3d,
    _hide_layer_if_present,
    _normalize_layer_selection,
    _replace_or_add_image,
    _replace_or_add_labels,
    _replace_or_add_points,
    _should_render_layer,
    _viewer_is_usable,
    extract_label_masks_from_viewer,
    show_analysis_results,
    show_optional_region_segmentation,
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
    def remove(self, layer):
        for key, value in list(self.items()):
            if value is layer:
                del self[key]
                return
        raise KeyError(layer)


class _FakeDims:
    def __init__(self):
        self.ndim = 3


class _FakeViewer:
    _instances: list["_FakeViewer"] = []

    def __init__(self):
        self.layers = _FakeLayerList()
        self.dims = _FakeDims()
        type(self)._instances.append(self)

    def add_image(self, data, name, **kwargs):
        self.layers[name] = _FakeLayer(data, **kwargs)
        return self.layers[name]

    def add_labels(self, data, name, **kwargs):
        self.layers[name] = _FakeLayer(data, **kwargs)
        return self.layers[name]

    def add_points(self, data, name, **kwargs):
        self.layers[name] = _FakeLayer(data, **kwargs)
        return self.layers[name]


def _make_loaded_images(tmp_path) -> LoadedImageChannels:
    source = tmp_path / "sample.ome.tif"
    source.write_bytes(b"fake")
    return LoadedImageChannels(
        source_path=source,
        paths=build_results_paths(source),
        voxel_scale_zyx=(1.0, 0.5, 0.5),
        cell_image=np.ones((1, 4, 4), dtype=np.float32),
        marker_image=np.ones((1, 4, 4), dtype=np.float32) * 2,
        optional_region_image=np.ones((1, 4, 4), dtype=np.float32) * 3,
        raw_shape_tzcyx=(1, 1, 3, 4, 4),
        raw_z_size=1,
        is_3d=False,
        metadata={},
    )


def _make_run_result() -> ColocalizationRunResult:
    tables = ColocalizationTables(
        detailed=pd.DataFrame([{"roi_id": 1, "cell_label": 1, "overlap_voxels": 1}]),
        summary=pd.DataFrame([{"roi_id": 1, "cell_label": 1, "marker_positive": True}]),
        overview=pd.DataFrame([{"roi_id": 1, "n_cells": 1}]),
    )
    return ColocalizationRunResult(
        cell_masks=np.ones((1, 4, 4), dtype=np.uint32),
        marker_masks=np.ones((1, 4, 4), dtype=np.uint32) * 2,
        positive_cell_masks=np.ones((1, 4, 4), dtype=np.uint32),
        optional_region_masks=np.ones((1, 4, 4), dtype=np.uint32) * 3,
        tables=tables,
        analysis_z_bounds=(0, 1),
    )


def test_visualization_helpers_update_fake_layers(tmp_path) -> None:
    viewer = _FakeViewer()
    assert _viewer_is_usable(viewer) is True
    assert _normalize_layer_selection(["a", " b "]) == {"a", "b"}
    assert _should_render_layer(None, "x") is True
    assert _should_render_layer({"x"}, "x") is True
    assert _should_render_layer({"x"}, "y") is False

    _replace_or_add_image(viewer, replace_existing_layers=True, name="img", data=np.zeros((1, 2, 2)), scale=(1, 1, 1))
    _replace_or_add_labels(viewer, replace_existing_layers=True, name="lab", data=np.ones((1, 2, 2), dtype=np.uint32), scale=(1, 1, 1))
    _replace_or_add_points(viewer, replace_existing_layers=True, name="pts", data=np.array([[1.0, 1.0]]), scale=(1, 1), size=5, text={"string": ["ROI 1"]})

    assert "img" in viewer.layers
    assert "lab" in viewer.layers
    assert "pts" in viewer.layers

    _hide_layer_if_present(viewer, "img")
    assert viewer.layers["img"].visible is False


def test_build_roi_labels_and_extract_masks() -> None:
    roi_labels_2d = np.array([[0, 1], [1, 0]], dtype=np.uint16)
    roi_labels_3d = _build_roi_labels_3d(roi_labels_2d, z_size=2, analysis_z_bounds=(1, 2))
    assert roi_labels_3d.shape == (2, 2, 2)
    assert np.count_nonzero(roi_labels_3d[0]) == 0
    assert np.count_nonzero(roi_labels_3d[1]) > 0

    viewer = _FakeViewer()
    viewer.add_labels(np.ones((1, 2, 2), dtype=np.uint32), name="Cellpose cell masks")
    viewer.add_labels(np.ones((1, 2, 2), dtype=np.uint32) * 2, name="Cellpose marker masks")
    cell_masks, marker_masks = extract_label_masks_from_viewer(viewer)
    assert set(np.unique(cell_masks)) == {1}
    assert set(np.unique(marker_masks)) == {2}

    with pytest.raises(KeyError):
        extract_label_masks_from_viewer(_FakeViewer())


def test_show_analysis_results_and_optional_region_segmentation(tmp_path) -> None:
    loaded = _make_loaded_images(tmp_path)
    run_result = _make_run_result()
    roi_labels = np.ones((4, 4), dtype=np.uint16)
    viewer = _FakeViewer()

    viewer = show_analysis_results(
        loaded_images=loaded,
        roi_labels_2d=roi_labels,
        run_result=run_result,
        display_names=DisplayNames(cell="Cell", marker="Marker", optional_region="Opt", positive_cells="Positive"),
        viewer=viewer,
        layers_to_show=["cell_image", "marker_image", "optional_region_image", "optional_region_labels", "rois", "roi_numbers", "cell_masks", "marker_masks", "positive_cells"],
        show_optional_region_image=True,
    )
    assert "Cell" in viewer.layers
    assert "Marker" in viewer.layers
    assert "Opt" in viewer.layers
    assert "Opt threshold labels" in viewer.layers
    assert "ROIs" in viewer.layers
    assert "ROI numbers" in viewer.layers
    assert "Cellpose cell masks" in viewer.layers
    assert "Cellpose marker masks" in viewer.layers
    assert "Positive" in viewer.layers

    region_result = OptionalRegionSegmentationResult(
        mask=np.ones((1, 4, 4), dtype=bool),
        labels=np.ones((1, 4, 4), dtype=np.uint32),
        threshold=1.0,
        corrected_image=np.ones((1, 4, 4), dtype=np.float32),
    )
    viewer = show_optional_region_segmentation(
        loaded_images=loaded,
        region_result=region_result,
        roi_labels_2d=roi_labels,
        display_names=DisplayNames(optional_region="Opt"),
        viewer=viewer,
    )
    assert "Opt background corrected" in viewer.layers
