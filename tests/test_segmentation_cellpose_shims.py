from __future__ import annotations

import numpy as np
import pytest

from cellcoloc import CellposeModelConfig
from cellcoloc.schemas import CellposeRefinementRoiCache
from cellcoloc.segmentation import (
    create_cellpose_model,
    create_cellpose_models_for_channels,
    evaluate_cellpose_model,
    evaluate_segmentation_method,
)


class _FakeCellposeModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.eval_calls = []
        self.compute_mask_calls = []

    def eval(self, image, **kwargs):
        self.eval_calls.append((image, kwargs))
        if kwargs.get("do_3D"):
            masks = np.ones((image.shape[0], image.shape[1], image.shape[2]), dtype=np.uint32)
            flows = [None, np.ones((3, image.shape[0], image.shape[1], image.shape[2])), np.ones((image.shape[0], image.shape[1], image.shape[2]))]
        else:
            masks = np.ones((image.shape[0], image.shape[1]), dtype=np.uint32)
            flows = [None, np.ones((2, image.shape[0], image.shape[1])), np.ones((image.shape[0], image.shape[1]))]
        return masks, flows, None

    def _compute_masks(self, shape_for_masks, dP, cellprob, **kwargs):
        self.compute_mask_calls.append((shape_for_masks, dP, cellprob, kwargs))
        if kwargs.get("do_3D"):
            return np.ones(shape_for_masks, dtype=np.uint32)
        return np.ones((shape_for_masks[1], shape_for_masks[2]), dtype=np.uint32)


def test_create_cellpose_model_and_shared_model_logic(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("cellcoloc.segmentation.models.CellposeModel", _FakeCellposeModel)
    monkeypatch.setattr("cellcoloc.segmentation.get_available_cellpose_model_names", lambda: ["cpsam", "cyto3"])
    monkeypatch.setattr("cellcoloc.segmentation.get_cellpose_major_version", lambda: 4)

    built_in = create_cellpose_model("cpsam", use_gpu=True)
    assert isinstance(built_in, _FakeCellposeModel)
    assert built_in.kwargs["gpu"] is True

    custom_path = tmp_path / "custom_model"
    custom_path.write_text("fake", encoding="utf-8")
    custom = create_cellpose_model(str(custom_path), use_gpu=False)
    assert custom.kwargs["pretrained_model"] == str(custom_path.resolve())

    shared_cell, shared_marker = create_cellpose_models_for_channels(
        CellposeModelConfig(model_name_or_path="cpsam", segmentation_method="cellpose"),
        CellposeModelConfig(model_name_or_path="cpsam", segmentation_method="cellpose"),
        use_gpu=False,
    )
    assert shared_cell is shared_marker


def test_evaluate_cellpose_model_for_v4_and_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeCellposeModel()
    image_2d = np.ones((1, 4, 4), dtype=np.float32)

    monkeypatch.setattr("cellcoloc.segmentation.get_cellpose_major_version", lambda: 4)
    cfg_v4 = CellposeModelConfig(model_name_or_path="cpsam", diameter=None, cellprob_threshold=0.1, flow_threshold=0.2)
    masks, refinement = evaluate_cellpose_model(model, image_2d, cfg_v4, (1.0, 1.0, 1.0))
    assert masks.shape == image_2d.shape
    assert isinstance(refinement, CellposeRefinementRoiCache)

    monkeypatch.setattr("cellcoloc.segmentation.get_cellpose_major_version", lambda: 3)
    cfg_v3 = CellposeModelConfig(model_name_or_path="cpsam", diameter=30)
    masks_v3, refinement_v3 = evaluate_cellpose_model(model, image_2d, cfg_v3, (1.0, 1.0, 1.0))
    assert masks_v3.shape == image_2d.shape
    assert refinement_v3 is None

    with pytest.raises(ValueError):
        evaluate_cellpose_model(model, image_2d, CellposeModelConfig(model_name_or_path="cpsam", diameter=None), (1.0, 1.0, 1.0))


def test_evaluate_segmentation_method_requires_model_for_cellpose() -> None:
    cfg = CellposeModelConfig(model_name_or_path="cpsam", segmentation_method="cellpose")
    with pytest.raises(ValueError):
        evaluate_segmentation_method(None, np.ones((1, 4, 4), dtype=np.float32), cfg, (1.0, 1.0, 1.0))
