"""Image prefiltering and label-mask postfiltering helpers.

This module keeps optional signal-processing steps out of the project-specific
user scripts. Prefilters operate on image intensities before Cellpose is run,
while postfilters remove suspicious label masks after segmentation by
comparing them against the original image intensities.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.measure import regionprops
from skimage.morphology import ball, binary_dilation, disk

from .config import CellposeModelConfig


def apply_prefilter(
    image_zyx: np.ndarray,
    model_config: CellposeModelConfig,
) -> np.ndarray:
    """Apply the configured optional prefilter to a ``ZYX`` image volume.

    Parameters
    ----------
    image_zyx:
        Input image as ``(Z, Y, X)``. True 2D data must be represented as a
        singleton-z volume ``(1, Y, X)``.
    model_config:
        Per-channel Cellpose configuration that carries the optional prefilter
        settings.

    Returns
    -------
    np.ndarray
        Filtered image with the same shape and dtype as a float array.
    """

    prefilter = _normalize_prefilter_name(model_config.prefilter)
    image_float = np.asarray(image_zyx, dtype=np.float32)
    is_3d = image_float.shape[0] > 1

    if prefilter is None:
        return image_float

    if prefilter == "gaussian":
        sigma_xy = model_config.prefilter_sigma_xy
        sigma_z = (
            model_config.prefilter_sigma_xy
            if model_config.prefilter_sigma_z is None
            else model_config.prefilter_sigma_z
        )
        if sigma_xy < 0 or sigma_z < 0:
            raise ValueError(
                "Gaussian prefilter sigmas must be non-negative, got "
                f"sigma_xy={sigma_xy}, sigma_z={sigma_z}."
            )
        if is_3d:
            return gaussian_filter(image_float, sigma=(sigma_z, sigma_xy, sigma_xy))
        return gaussian_filter(image_float[0], sigma=(sigma_xy, sigma_xy))[np.newaxis, :, :]

    if prefilter == "median":
        size_xy = model_config.prefilter_median_size_xy
        size_z = (
            model_config.prefilter_median_size_xy
            if model_config.prefilter_median_size_z is None
            else model_config.prefilter_median_size_z
        )
        if size_xy < 1 or size_z < 1:
            raise ValueError(
                "Median prefilter sizes must be positive integers, got "
                f"size_xy={size_xy}, size_z={size_z}."
            )
        if is_3d:
            return median_filter(image_float, size=(size_z, size_xy, size_xy))
        return median_filter(image_float[0], size=(size_xy, size_xy))[np.newaxis, :, :]

    raise ValueError(f"Unsupported prefilter option: {model_config.prefilter!r}.")


def apply_postfilters(
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    model_config: CellposeModelConfig,
) -> np.ndarray:
    """Apply the configured optional postfilters to a label image.

    The filters are executed sequentially in the order given by
    ``model_config.postfilters``. Each filter removes labels that do not pass
    its criterion and leaves the surviving label ids unchanged.
    """

    postfilters = _normalize_postfilters(model_config.postfilters)
    filtered = np.asarray(label_image, dtype=np.uint32).copy()
    original = np.asarray(intensity_image, dtype=np.float32)

    for postfilter in postfilters:
        if postfilter == "min_intensity":
            filtered = _apply_min_intensity_postfilter(filtered, original, model_config)
        elif postfilter == "local_contrast":
            filtered = _apply_local_contrast_postfilter(filtered, original, model_config)
        else:
            raise ValueError(f"Unsupported postfilter option: {postfilter!r}.")

    return filtered


def _normalize_prefilter_name(prefilter: str | None) -> str | None:
    """Return a normalized prefilter name or ``None`` when disabled."""

    if prefilter is None:
        return None
    normalized = prefilter.strip().lower()
    if normalized in {"gaussian", "median"}:
        return normalized
    raise ValueError(
        "`CellposeModelConfig.prefilter` must be None, 'gaussian', or "
        f"'median', got {prefilter!r}."
    )


def _normalize_postfilters(postfilters: str | Sequence[str] | None) -> list[str]:
    """Normalize postfilter configuration to a lowercase list."""

    if postfilters is None:
        return []
    if isinstance(postfilters, str):
        values = [postfilters]
    else:
        values = list(postfilters)

    normalized = [value.strip().lower() for value in values]
    allowed = {"min_intensity", "local_contrast"}
    invalid = [value for value in normalized if value not in allowed]
    if invalid:
        raise ValueError(
            "`CellposeModelConfig.postfilters` contains unsupported values: "
            f"{invalid}. Allowed values are {sorted(allowed)}."
        )
    return normalized


def _apply_min_intensity_postfilter(
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    model_config: CellposeModelConfig,
) -> np.ndarray:
    """Remove labels that do not reach the configured minimum intensity."""

    threshold = model_config.min_intensity_threshold
    if threshold is None:
        raise ValueError(
            "`min_intensity` postfilter requires "
            "`CellposeModelConfig.min_intensity_threshold` to be set."
        )

    measure = model_config.min_intensity_measure.strip().lower()
    if measure not in {"mean", "max"}:
        raise ValueError(
            "`CellposeModelConfig.min_intensity_measure` must be 'mean' or "
            f"'max', got {model_config.min_intensity_measure!r}."
        )

    filtered = label_image.copy()
    for label_value in np.unique(filtered):
        if label_value == 0:
            continue
        label_mask = filtered == label_value
        intensities = intensity_image[label_mask]
        if intensities.size == 0:
            filtered[label_mask] = 0
            continue

        score = float(intensities.mean()) if measure == "mean" else float(intensities.max())
        if score <= threshold:
            filtered[label_mask] = 0

    return filtered


def _apply_local_contrast_postfilter(
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    model_config: CellposeModelConfig,
) -> np.ndarray:
    """Remove labels that are not brighter than their local surrounding shell."""

    inner_radius = model_config.local_contrast_shell_inner_radius
    outer_radius = model_config.local_contrast_shell_outer_radius
    k_value = model_config.local_contrast_k

    if inner_radius < 0 or outer_radius <= inner_radius:
        raise ValueError(
            "Local-contrast shell radii must satisfy "
            f"0 <= inner < outer, got inner={inner_radius}, outer={outer_radius}."
        )

    filtered = label_image.copy()
    is_3d = filtered.shape[0] > 1

    for region in regionprops(filtered):
        label_value = int(region.label)
        slices = region.slice
        label_crop = filtered[slices] == label_value
        intensity_crop = intensity_image[slices]
        labels_crop = filtered[slices]

        if is_3d:
            shell_mask = _build_3d_shell(label_crop, inner_radius, outer_radius)
        else:
            shell_2d = _build_2d_shell(label_crop[0], inner_radius, outer_radius)
            shell_mask = shell_2d[np.newaxis, :, :]

        shell_mask &= ~label_crop
        shell_mask &= labels_crop == 0
        if not shell_mask.any():
            continue

        object_values = intensity_crop[label_crop]
        shell_values = intensity_crop[shell_mask]
        if object_values.size == 0 or shell_values.size == 0:
            continue

        object_median = float(np.median(object_values))
        background_median = float(np.median(shell_values))
        background_mad = float(np.median(np.abs(shell_values - background_median)))

        if object_median <= background_median + (k_value * background_mad):
            filtered[filtered == label_value] = 0

    return filtered


def _build_2d_shell(
    mask_yx: np.ndarray,
    inner_radius: int,
    outer_radius: int,
) -> np.ndarray:
    """Build a 2D shell around one binary object mask."""

    outer = binary_dilation(mask_yx, footprint=disk(outer_radius))
    if inner_radius > 0:
        inner = binary_dilation(mask_yx, footprint=disk(inner_radius))
    else:
        inner = mask_yx
    return outer & ~inner


def _build_3d_shell(
    mask_zyx: np.ndarray,
    inner_radius: int,
    outer_radius: int,
) -> np.ndarray:
    """Build a 3D shell around one binary object mask."""

    outer = binary_dilation(mask_zyx, footprint=ball(outer_radius))
    if inner_radius > 0:
        inner = binary_dilation(mask_zyx, footprint=ball(inner_radius))
    else:
        inner = mask_zyx
    return outer & ~inner
