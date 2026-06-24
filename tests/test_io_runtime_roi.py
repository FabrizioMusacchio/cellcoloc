from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellcoloc import ChannelConfig, ColocalizationRunResult
from cellcoloc.io import (
    _convert_length_to_microns,
    _extract_zyx_channel,
    _resolve_voxel_scale_zyx,
    build_results_paths,
    export_analysis_outputs,
    load_analysis_images,
    load_roi_labels,
    save_roi_labels,
    try_load_roi_labels,
)
from cellcoloc.roi import (
    create_full_image_roi_labels,
    get_bbox_2d,
    get_roi_label_points,
    rasterize_shapes_to_labelmask,
)
from cellcoloc.runtime import (
    _ensure_directory_and_probe_write,
    _set_env_if_fallback_needed,
    get_runtime_cache_root,
)
from cellcoloc.schemas import ColocalizationTables, ResultsPaths


class _DummyShapesLayer:
    def __init__(self, data: list[np.ndarray]) -> None:
        self.data = data


def _make_results_paths(tmp_path: Path) -> ResultsPaths:
    source = tmp_path / "sample.ome.tif"
    source.write_bytes(b"fake")
    return build_results_paths(source)


def test_convert_length_to_microns_supports_common_units() -> None:
    assert _convert_length_to_microns(1.0, "micron", "X") == 1.0
    assert _convert_length_to_microns(1000.0, "nm", "X") == 1.0
    assert _convert_length_to_microns(0.001, "mm", "X") == 1.0


def test_convert_length_to_microns_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        _convert_length_to_microns(0.0, "micron", "X")
    with pytest.raises(ValueError):
        _convert_length_to_microns(1.0, "parsec", "X")


def test_resolve_voxel_scale_zyx_supports_yx_input_and_metadata_fallback() -> None:
    metadata = {
        "PhysicalSizeZ": 2.0,
        "PhysicalSizeY": 0.5,
        "PhysicalSizeX": 0.25,
        "PhysicalSizeZUnit": "micron",
        "PhysicalSizeYUnit": "micron",
        "PhysicalSizeXUnit": "micron",
    }
    assert _resolve_voxel_scale_zyx((0.5, 0.25), metadata) == (1.0, 0.5, 0.25)
    assert _resolve_voxel_scale_zyx(None, metadata) == (2.0, 0.5, 0.25)
    assert _resolve_voxel_scale_zyx(None, {}) == (1.0, 1.0, 1.0)


def test_extract_zyx_channel_normalizes_single_plane_to_singleton_z() -> None:
    image = np.zeros((1, 1, 2, 4, 4), dtype=np.uint16)
    image[0, 0, 1, 1, 1] = 7
    channel = _extract_zyx_channel(image, 1)
    assert channel.shape == (1, 4, 4)
    assert channel[0, 1, 1] == 7


def test_load_analysis_images_uses_monkeypatched_omio_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image = np.zeros((1, 2, 3, 4, 5), dtype=np.uint16)
    image[0, 0, 0, 1, 1] = 10
    image[0, 0, 1, 2, 2] = 20
    image[0, 0, 2, 3, 3] = 30
    metadata = {
        "PhysicalSizeZ": 2.0,
        "PhysicalSizeY": 0.5,
        "PhysicalSizeX": 0.25,
        "PhysicalSizeZUnit": "micron",
        "PhysicalSizeYUnit": "micron",
        "PhysicalSizeXUnit": "micron",
    }

    def fake_imread(path: Path, zarr_store=None, reuse_disk_cache=False):
        return image, metadata

    monkeypatch.setattr("cellcoloc.io.om.imread", fake_imread)
    source = tmp_path / "sample.ome.tif"
    source.write_bytes(b"fake")

    loaded = load_analysis_images(
        source_path=source,
        channel_config=ChannelConfig(cell_channel=0, marker_channel=1, optional_region_channel=2),
        voxel_scale_zyx=None,
        image_loading_mode="memory",
    )

    assert loaded.cell_image.shape == (2, 4, 5)
    assert loaded.marker_image.shape == (2, 4, 5)
    assert loaded.optional_region_image is not None
    assert loaded.voxel_scale_zyx == (2.0, 0.5, 0.25)


def test_save_load_and_try_load_roi_labels_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "roi.tif"
    labels = np.array([[0, 1], [2, 2]], dtype=np.uint16)
    save_roi_labels(path, labels)
    loaded = load_roi_labels(path)
    assert np.array_equal(loaded, labels)
    assert np.array_equal(try_load_roi_labels(path), labels)
    assert try_load_roi_labels(tmp_path / "missing.tif") is None


def test_export_analysis_outputs_writes_expected_files(tmp_path: Path) -> None:
    paths = _make_results_paths(tmp_path)
    detailed = pd.DataFrame([{"roi_id": 1, "cell_label": 1, "overlap_voxels": 2}])
    summary = pd.DataFrame(
        [
            {
                "roi_id": 1,
                "cell_label": 1,
                "marker_positive": True,
                "cell_area_px_2d": 4.0,
            }
        ]
    )
    overview = pd.DataFrame([{"roi_id": 1, "n_cells": 1}])
    marker_properties = pd.DataFrame([{"roi_id": 1, "marker_label": 1, "marker_area_px_2d": 4.0}])
    third_channel_properties = pd.DataFrame(
        [{"roi_id": 1, "optional_region_label": 1, "optional_region_area_px_2d": 4.0}]
    )
    roi_cell_summary = pd.DataFrame([{"roi_id": 1, "n_cells": 1, "average_cell_area_px_2d": 4.0}])
    roi_marker_summary = pd.DataFrame(
        [{"roi_id": 1, "n_marker_objects": 1, "average_marker_area_px_2d": 4.0}]
    )
    roi_third_channel_summary = pd.DataFrame(
        [{"roi_id": 1, "n_3rd_channel_objects": 1, "average_optional_region_area_px_2d": 4.0}]
    )
    run_result = ColocalizationRunResult(
        cell_masks=np.ones((1, 2, 2), dtype=np.uint32),
        marker_masks=np.ones((1, 2, 2), dtype=np.uint32),
        positive_cell_masks=np.ones((1, 2, 2), dtype=np.uint32),
        optional_region_masks=np.ones((1, 2, 2), dtype=np.uint32),
        tables=ColocalizationTables(
            detailed=detailed,
            summary=summary,
            overview=overview,
            marker_properties=marker_properties,
            third_channel_properties=third_channel_properties,
            roi_cell_summary=roi_cell_summary,
            roi_marker_summary=roi_marker_summary,
            roi_third_channel_summary=roi_third_channel_summary,
        ),
    )

    export_analysis_outputs(run_result, paths)

    assert paths.detailed_csv_path.exists()
    assert paths.excel_path.exists()
    assert paths.cell_mask_path.exists()
    assert paths.marker_mask_path.exists()
    assert paths.positive_cell_mask_path.exists()
    assert paths.optional_region_mask_path.exists()
    workbook = pd.ExcelFile(paths.excel_path)
    assert workbook.sheet_names == [
        "detailed_overlaps",
        "cell_summary",
        "marker_properties",
        "3rd_channel_properties",
        "roi_coloc_overview",
        "roi_cell_summary",
        "roi_marker_summary",
        "roi_3rd_channel_summary",
    ]


def test_runtime_helpers_create_and_use_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert get_runtime_cache_root().name == "cellcoloc_runtime_cache"
    assert _ensure_directory_and_probe_write(tmp_path / "writable")

    def fake_probe(directory: Path) -> bool:
        return False

    monkeypatch.setattr("cellcoloc.runtime._ensure_directory_and_probe_write", fake_probe)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    changed = _set_env_if_fallback_needed("MPLCONFIGDIR", tmp_path / "blocked")

    assert changed is True
    assert "MPLCONFIGDIR" in os.environ


def test_roi_helpers_cover_rasterization_and_centroids() -> None:
    triangle = np.array([[1.0, 1.0], [1.0, 4.0], [4.0, 1.0]])
    labels = rasterize_shapes_to_labelmask(_DummyShapesLayer([triangle]), (6, 6))
    assert labels.max() == 1

    bbox = get_bbox_2d(labels == 1)
    assert bbox is not None
    assert bbox[0].start <= bbox[0].stop

    points, text = get_roi_label_points(labels)
    assert points.shape == (1, 2)
    assert text == ["ROI 1"]

    full = create_full_image_roi_labels((3, 4))
    assert full.shape == (3, 4)
    assert np.all(full == 1)
