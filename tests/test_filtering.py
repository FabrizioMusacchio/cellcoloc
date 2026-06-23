from __future__ import annotations

import numpy as np
import pytest

from cellcoloc import CellposeModelConfig
from cellcoloc.filtering import (
    _build_2d_shell,
    _build_3d_shell,
    _normalize_postfilters,
    _normalize_prefilters,
    apply_postfilters,
    apply_prefilter,
)


def test_normalize_prefilters_accepts_aliases_and_sequences() -> None:
    assert _normalize_prefilters(None) == []
    assert _normalize_prefilters("Gaussian") == ["gaussian"]
    assert _normalize_prefilters(["median", "log"]) == ["median", "laplacian_of_gaussian"]


def test_normalize_prefilters_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        _normalize_prefilters(["unknown"])


def test_apply_prefilter_preserves_shape_for_2d_and_3d() -> None:
    image_2d = np.zeros((1, 5, 5), dtype=np.float32)
    image_2d[0, 2, 2] = 10
    image_3d = np.zeros((3, 5, 5), dtype=np.float32)
    image_3d[1, 2, 2] = 10

    cfg = CellposeModelConfig(
        model_name_or_path="cpsam",
        prefilter=["median", "gaussian"],
        prefilter_sigma_xy=0.5,
        prefilter_sigma_z=0.25,
        prefilter_median_size_xy=3,
        prefilter_median_size_z=3,
    )

    filtered_2d = apply_prefilter(image_2d, cfg)
    filtered_3d = apply_prefilter(image_3d, cfg)

    assert filtered_2d.shape == image_2d.shape
    assert filtered_3d.shape == image_3d.shape
    assert filtered_2d.dtype == np.float32
    assert filtered_3d.dtype == np.float32


def test_apply_prefilter_rejects_negative_gaussian_sigma() -> None:
    cfg = CellposeModelConfig(
        model_name_or_path="cpsam",
        prefilter="gaussian",
        prefilter_sigma_xy=-1.0,
    )
    with pytest.raises(ValueError):
        apply_prefilter(np.zeros((1, 3, 3), dtype=np.float32), cfg)


def test_apply_postfilters_min_intensity_max_removes_weak_label() -> None:
    labels = np.array([[[1, 1, 0], [0, 2, 2], [0, 0, 0]]], dtype=np.uint32)
    image = np.array([[[10, 10, 0], [0, 100, 120], [0, 0, 0]]], dtype=np.float32)
    cfg = CellposeModelConfig(
        model_name_or_path="cpsam",
        postfilters="min_intensity",
        min_intensity_measure="max",
        min_intensity_threshold=50,
    )

    filtered = apply_postfilters(labels, image, cfg)

    assert 1 not in np.unique(filtered)
    assert 2 in np.unique(filtered)


def test_apply_postfilters_bright_pixel_fraction_removes_sparse_object() -> None:
    labels = np.array([[[1, 1, 0], [0, 2, 2], [0, 0, 0]]], dtype=np.uint32)
    image = np.array([[[100, 0, 0], [0, 200, 200], [0, 0, 0]]], dtype=np.float32)
    cfg = CellposeModelConfig(
        model_name_or_path="cpsam",
        postfilters="bright_pixel_support",
        bright_pixel_measure="fraction",
        bright_pixel_threshold=50,
        bright_pixel_min_fraction=0.75,
    )

    filtered = apply_postfilters(labels, image, cfg)

    assert 1 not in np.unique(filtered)
    assert 2 in np.unique(filtered)


def test_apply_postfilters_local_contrast_keeps_bright_object_and_removes_dim_object() -> None:
    labels = np.zeros((1, 9, 9), dtype=np.uint32)
    labels[0, 4, 4] = 1
    labels[0, 0:3, 0] = 2
    labels[0, 2, 0:3] = 2
    image = np.zeros((1, 9, 9), dtype=np.float32)
    image[0, 4, 4] = 50
    image[0, 0:3, 0:3] = 5
    image[0, 0:3, 0] = 1
    image[0, 2, 0:3] = 1

    cfg = CellposeModelConfig(
        model_name_or_path="cpsam",
        postfilters="local_contrast",
        local_contrast_k=1.0,
        local_contrast_shell_inner_radius=1,
        local_contrast_shell_outer_radius=2,
    )

    filtered = apply_postfilters(labels, image, cfg)

    assert 1 in np.unique(filtered)
    assert 2 not in np.unique(filtered)


def test_normalize_postfilters_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        _normalize_postfilters(["not-a-filter"])


def test_shell_builders_create_nonempty_shells() -> None:
    mask_2d = np.zeros((5, 5), dtype=bool)
    mask_2d[2, 2] = True
    shell_2d = _build_2d_shell(mask_2d, inner_radius=0, outer_radius=2)
    assert shell_2d.any()
    assert not shell_2d[2, 2]

    mask_3d = np.zeros((3, 5, 5), dtype=bool)
    mask_3d[1, 2, 2] = True
    shell_3d = _build_3d_shell(mask_3d, inner_radius=0, outer_radius=1)
    assert shell_3d.any()
    assert not shell_3d[1, 2, 2]
