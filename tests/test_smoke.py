"""Small smoke tests for the public CellColoc package surface."""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import cellcoloc
from cellcoloc import (
    CellposeModelConfig,
    ColocalizationConfig,
    RuntimeConfig,
    SingleChannelAnalysisConfig,
    SingleChannelConfig,
)
from cellcoloc.segmentation import normalize_segmentation_method


def test_public_api_exports_expected_symbols() -> None:
    """The top-level package should expose the main user-facing configs."""

    assert hasattr(cellcoloc, "CellposeModelConfig")
    assert hasattr(cellcoloc, "RuntimeConfig")
    assert hasattr(cellcoloc, "run_roi_cellpose_colocalization")
    assert hasattr(cellcoloc, "run_roi_single_channel_segmentation")


def test_main_config_objects_can_be_instantiated() -> None:
    """Basic dataclass configs should be constructible with minimal settings."""

    model_config = CellposeModelConfig(model_name_or_path="cpsam")
    runtime_config = RuntimeConfig()
    coloc_config = ColocalizationConfig()
    single_channel_config = SingleChannelConfig(channel_index=0)
    single_analysis_config = SingleChannelAnalysisConfig()

    assert model_config.segmentation_method == "cellpose"
    assert runtime_config.use_gpu is True
    assert coloc_config.min_overlap_voxels > 0
    assert single_channel_config.channel_index == 0
    assert single_analysis_config.min_object_voxels > 0


def test_supported_segmentation_methods_normalize() -> None:
    """Supported segmentation method names should normalize correctly."""

    assert normalize_segmentation_method("cellpose") == "cellpose"
    assert normalize_segmentation_method("OTSU") == "otsu"
    assert normalize_segmentation_method("li") == "li"
    assert normalize_segmentation_method(" percentile ") == "percentile"
