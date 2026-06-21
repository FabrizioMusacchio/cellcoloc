"""Image prefiltering and label-mask postfiltering helpers.

This module keeps optional signal-processing steps out of the project-specific
user scripts. Prefilters operate on image intensities before Cellpose is run,
while postfilters remove suspicious label masks after segmentation by
comparing them against the original image intensities.

author: Fabrizio Musacchio
date: May/June 2026
"""
# %% IMPORTS
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace, median_filter
from skimage.measure import regionprops
from skimage.morphology import ball, dilation, disk

from .config import CellposeModelConfig

# %%FILTERING HELPERS
def apply_prefilter(
    image_zyx: np.ndarray,
    model_config: CellposeModelConfig,
) -> np.ndarray:
    """Apply the configured optional prefilter chain to a ``ZYX`` image volume.

    Parameters
    ----------
    image_zyx:
        Input image as ``(Z, Y, X)``. True 2D data must be represented as a
        singleton-z volume ``(1, Y, X)``.
    model_config:
        Per-channel Cellpose configuration that carries the optional prefilter
        settings. ``model_config.prefilter`` may be ``None``, one filter name,
        or a sequence of filter names executed in order.

    Returns
    -------
    np.ndarray
        Filtered image with the same shape and dtype as a float array.
    """

    image_float = np.asarray(image_zyx, dtype=np.float32)
    is_3d = image_float.shape[0] > 1

    prefilters = _normalize_prefilters(model_config.prefilter)
    filtered = image_float

    for prefilter in prefilters:
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
                filtered = gaussian_filter(filtered, sigma=(sigma_z, sigma_xy, sigma_xy))
            else:
                filtered = gaussian_filter(filtered[0], sigma=(sigma_xy, sigma_xy))[np.newaxis, :, :]
            continue

        if prefilter == "laplacian_of_gaussian":
            sigma_xy = model_config.prefilter_sigma_xy
            sigma_z = (
                model_config.prefilter_sigma_xy
                if model_config.prefilter_sigma_z is None
                else model_config.prefilter_sigma_z
            )
            if sigma_xy < 0 or sigma_z < 0:
                raise ValueError(
                    "Laplacian-of-Gaussian prefilter sigmas must be non-negative, "
                    f"got sigma_xy={sigma_xy}, sigma_z={sigma_z}."
                )

            # We invert the LoG response so that bright blob-like structures remain
            # positive and intuitive for downstream Cellpose normalization.
            if is_3d:
                filtered = -gaussian_laplace(
                    filtered,
                    sigma=(sigma_z, sigma_xy, sigma_xy),
                )
            else:
                filtered_2d = -gaussian_laplace(
                    filtered[0],
                    sigma=(sigma_xy, sigma_xy),
                )
                filtered = filtered_2d[np.newaxis, :, :]
            continue

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
                filtered = median_filter(filtered, size=(size_z, size_xy, size_xy))
            else:
                filtered = median_filter(filtered[0], size=(size_xy, size_xy))[np.newaxis, :, :]
            continue

        raise ValueError(f"Unsupported prefilter option: {prefilter!r}.")

    return filtered


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
        elif postfilter == "bright_pixel_support":
            filtered = _apply_bright_pixel_support_postfilter(filtered, original, model_config)
        else:
            raise ValueError(f"Unsupported postfilter option: {postfilter!r}.")

    return filtered


def _normalize_prefilters(prefilters: str | Sequence[str] | None) -> list[str]:
    """Normalize prefilter configuration to a lowercase ordered list."""

    if prefilters is None:
        return []
    if isinstance(prefilters, str):
        values = [prefilters]
    else:
        values = list(prefilters)

    normalized: list[str] = []
    for value in values:
        normalized_value = value.strip().lower()
        if normalized_value in {"gaussian", "median", "laplacian_of_gaussian", "log"}:
            normalized.append(
                "laplacian_of_gaussian" if normalized_value == "log" else normalized_value
            )
            continue
        raise ValueError(
            "`CellposeModelConfig.prefilter` must contain only None, "
            "'gaussian', 'laplacian_of_gaussian'/'log', or 'median', "
            f"got {value!r}."
        )

    return normalized


def _normalize_postfilters(postfilters: str | Sequence[str] | None) -> list[str]:
    """Normalize postfilter configuration to a lowercase list."""

    if postfilters is None:
        return []
    if isinstance(postfilters, str):
        values = [postfilters]
    else:
        values = list(postfilters)

    normalized = [value.strip().lower() for value in values]
    allowed = {"min_intensity", "local_contrast", "bright_pixel_support"}
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
    if measure not in {"mean", "median", "max"}:
        raise ValueError(
            "`CellposeModelConfig.min_intensity_measure` must be 'mean', "
            f"'median', or 'max', got {model_config.min_intensity_measure!r}."
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

        if measure == "mean":
            score = float(intensities.mean())
        elif measure == "median":
            score = float(np.median(intensities))
        else:
            score = float(intensities.max())
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


def _apply_bright_pixel_support_postfilter(
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    model_config: CellposeModelConfig,
) -> np.ndarray:
    """Remove labels with too little clearly bright signal support."""

    threshold = model_config.bright_pixel_threshold
    if threshold is None:
        raise ValueError(
            "`bright_pixel_support` postfilter requires "
            "`CellposeModelConfig.bright_pixel_threshold` to be set."
        )

    measure = model_config.bright_pixel_measure.strip().lower()
    if measure not in {"count", "fraction"}:
        raise ValueError(
            "`CellposeModelConfig.bright_pixel_measure` must be 'count' or "
            f"'fraction', got {model_config.bright_pixel_measure!r}."
        )

    if measure == "count":
        min_count = model_config.bright_pixel_min_count
        if min_count is None:
            raise ValueError(
                "`bright_pixel_support` with `bright_pixel_measure='count'` "
                "requires `CellposeModelConfig.bright_pixel_min_count` to be set."
            )
        if min_count < 0:
            raise ValueError(
                "`CellposeModelConfig.bright_pixel_min_count` must be >= 0, "
                f"got {min_count}."
            )
    else:
        min_fraction = model_config.bright_pixel_min_fraction
        if min_fraction is None:
            raise ValueError(
                "`bright_pixel_support` with `bright_pixel_measure='fraction'` "
                "requires `CellposeModelConfig.bright_pixel_min_fraction` to be set."
            )
        if min_fraction < 0 or min_fraction > 1:
            raise ValueError(
                "`CellposeModelConfig.bright_pixel_min_fraction` must be "
                f"between 0 and 1, got {min_fraction}."
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

        bright_pixel_count = int(np.count_nonzero(intensities > threshold))
        if measure == "count":
            keep_label = bright_pixel_count >= int(model_config.bright_pixel_min_count)
        else:
            bright_pixel_fraction = bright_pixel_count / float(intensities.size)
            keep_label = bright_pixel_fraction >= float(model_config.bright_pixel_min_fraction)

        if not keep_label:
            filtered[label_mask] = 0

    return filtered


def _build_2d_shell(
    mask_yx: np.ndarray,
    inner_radius: int,
    outer_radius: int,
) -> np.ndarray:
    """Build a 2D shell around one binary object mask."""

    outer = dilation(mask_yx, footprint=disk(outer_radius))
    if inner_radius > 0:
        inner = dilation(mask_yx, footprint=disk(inner_radius))
    else:
        inner = mask_yx
    return outer & ~inner


def _build_3d_shell(
    mask_zyx: np.ndarray,
    inner_radius: int,
    outer_radius: int,
) -> np.ndarray:
    """Build a 3D shell around one binary object mask."""

    outer = dilation(mask_zyx, footprint=ball(outer_radius))
    if inner_radius > 0:
        inner = dilation(mask_zyx, footprint=ball(inner_radius))
    else:
        inner = mask_zyx
    return outer & ~inner
# %% END
