"""Segmentation utilities used by the reusable colocalization pipeline."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from cellpose import models
from skimage.filters import gaussian, threshold_li, threshold_otsu
from skimage.measure import label
from skimage.morphology import ball, closing, remove_small_holes, remove_small_objects

from .config import CellposeModelConfig, OptionalRegionSegmentationConfig
from .schemas import OptionalRegionSegmentationResult


def get_cellpose_major_version() -> int | None:
    """Return the installed Cellpose major version when it can be determined."""

    try:
        raw_version = version("cellpose")
    except PackageNotFoundError:
        return None

    major_token = raw_version.split(".", maxsplit=1)[0]
    try:
        return int(major_token)
    except ValueError:
        return None


def get_available_cellpose_model_names() -> list[str]:
    """Return locally available built-in and user-registered Cellpose models."""

    available_models = set(getattr(models, "MODEL_NAMES", []))
    get_user_models = getattr(models, "get_user_models", None)
    if callable(get_user_models):
        try:
            available_models.update(get_user_models())
        except Exception:
            pass

    return sorted(model_name for model_name in available_models if model_name)


def create_cellpose_model(model_name_or_path: str, use_gpu: bool) -> models.CellposeModel:
    """Create a Cellpose model from either a built-in name or a custom path.

    The helper is intentionally strict: if a requested built-in model name is
    not locally available in the installed Cellpose version, it raises a clear
    error instead of silently falling back to a different model. It also adapts
    model construction to the installed Cellpose major version so that newer
    Cellpose 4 setups do not receive the deprecated ``model_type`` argument.
    """

    candidate_path = Path(model_name_or_path).expanduser()
    if candidate_path.exists():
        print(f"Loading Cellpose custom model from:\n{candidate_path.resolve()}")
        return models.CellposeModel(
            gpu=use_gpu,
            pretrained_model=str(candidate_path.resolve()),
        )

    available_models = get_available_cellpose_model_names()
    if model_name_or_path not in available_models:
        raise ValueError(
            "The requested Cellpose model is not available in the current "
            f"environment: {model_name_or_path!r}. Available model names: "
            f"{available_models or ['<none>']}. Please provide a local custom "
            "model path or use a Cellpose installation that exposes the "
            "required built-in model."
        )

    cellpose_major = get_cellpose_major_version()
    print(f"Loading built-in Cellpose model: {model_name_or_path}")
    if cellpose_major is not None and cellpose_major >= 4:
        if model_name_or_path == "cpsam":
            return models.CellposeModel(gpu=use_gpu)
        return models.CellposeModel(
            gpu=use_gpu,
            pretrained_model=model_name_or_path,
        )

    return models.CellposeModel(
        gpu=use_gpu,
        pretrained_model=model_name_or_path,
        model_type=model_name_or_path,
    )


def evaluate_cellpose_model(
    model: models.CellposeModel,
    image_zyx: np.ndarray,
    model_config: CellposeModelConfig,
) -> np.ndarray:
    """Run Cellpose and return the resulting label image as ``uint32``.

    The function accepts both 3D ``ZYX`` arrays and 2D images represented as a
    singleton-z ``(1, Y, X)`` volume. When ``model_config.do_3d`` is ``None``,
    the function auto-detects the correct Cellpose mode from the z-size.
    """

    do_3d = model_config.do_3d
    if do_3d is None:
        do_3d = image_zyx.shape[0] > 1

    if not do_3d and image_zyx.shape[0] != 1:
        raise ValueError(
            "A 2D Cellpose run was requested for an image with more than one z "
            "slice. Please keep automatic 3D detection enabled or project the "
            "image to 2D before segmentation."
        )

    cellpose_input = image_zyx if do_3d else image_zyx[0]

    masks, _, _ = model.eval(
        cellpose_input,
        do_3D=do_3d,
        z_axis=model_config.z_axis if do_3d else None,
        channel_axis=model_config.channel_axis,
        diameter=model_config.diameter,
    )

    masks_array = np.asarray(masks, dtype=np.uint32)
    if do_3d:
        return masks_array

    if masks_array.ndim != 2:
        raise ValueError(
            "Cellpose returned an unexpected 2D mask shape: "
            f"{masks_array.shape}."
        )

    return masks_array[np.newaxis, :, :]


def relabel_with_offset(mask: np.ndarray, offset: int) -> np.ndarray:
    """Shift all non-zero labels by a fixed offset."""

    out = mask.copy()
    valid = out > 0
    out[valid] += offset
    return out


def filter_labels_by_size(label_image: np.ndarray, min_size: int) -> np.ndarray:
    """Remove labels smaller than the configured voxel threshold."""

    labels, counts = np.unique(label_image, return_counts=True)
    keep_labels = labels[(labels != 0) & (counts >= min_size)]
    lookup = np.zeros(int(label_image.max()) + 1, dtype=label_image.dtype)
    lookup[keep_labels] = keep_labels
    return lookup[label_image]


def _legacy_threshold_to_max_size(value: int) -> int:
    """Convert the old strict threshold semantics to the new scikit-image API."""

    return max(int(value) - 1, 0)


def segment_optional_region(
    image_zyx: np.ndarray,
    roi_labels_2d: np.ndarray | None,
    config: OptionalRegionSegmentationConfig,
) -> OptionalRegionSegmentationResult:
    """Threshold an optional third channel and compute a cleaned 3D mask.

    The function keeps the old prototype behavior while avoiding the deprecated
    scikit-image arguments by mapping strict minimum thresholds to the new
    inclusive ``max_size`` semantics.
    """

    print("\nSegmenting optional region channel...")
    image_float = image_zyx.astype(np.float32, copy=False)

    roi_mask_3d = None
    if roi_labels_2d is not None:
        roi_mask_3d = np.repeat((roi_labels_2d > 0)[np.newaxis, :, :], image_float.shape[0], axis=0)

    image_work = image_float.copy()
    if roi_mask_3d is not None:
        image_work[~roi_mask_3d] = 0

    if config.gaussian_sigma is not None and config.gaussian_sigma > 0:
        print(f"    Optional region: Gaussian smoothing with sigma={config.gaussian_sigma}...")
        image_smooth = gaussian(image_work, sigma=config.gaussian_sigma, preserve_range=True)
        if roi_mask_3d is not None:
            image_smooth[~roi_mask_3d] = 0
    else:
        image_smooth = image_work

    if config.background_sigma is not None and config.background_sigma > 0:
        print(f"    Optional region: background subtraction with sigma={config.background_sigma}...")
        background = gaussian(image_smooth, sigma=config.background_sigma, preserve_range=True)
        image_corrected = image_smooth - background
        image_corrected[image_corrected < 0] = 0
        if roi_mask_3d is not None:
            image_corrected[~roi_mask_3d] = 0
    else:
        image_corrected = image_smooth

    if roi_mask_3d is not None:
        values = image_corrected[roi_mask_3d]
    else:
        values = image_corrected.ravel()

    values = values[np.isfinite(values)]
    values = values[values > 0]

    if len(values) == 0:
        raise ValueError("Cannot compute an optional-region threshold because no positive values were found.")

    method = config.method.lower()
    print(f"    Optional region: computing threshold with method='{method}'...")
    if method == "otsu":
        threshold = float(threshold_otsu(values))
    elif method == "li":
        threshold = float(threshold_li(values))
    elif method == "percentile":
        threshold = float(np.percentile(values, config.percentile))
    else:
        raise ValueError(f"Unknown optional region segmentation method: {config.method}")

    if roi_mask_3d is not None:
        region_mask = np.zeros_like(image_corrected, dtype=bool)
        region_mask[roi_mask_3d] = image_corrected[roi_mask_3d] > threshold
    else:
        region_mask = image_corrected > threshold

    if config.apply_closing:
        print("    Optional region: applying morphological closing...")
        region_mask = closing(region_mask, footprint=ball(1))
        if roi_mask_3d is not None:
            region_mask[~roi_mask_3d] = False

    if config.min_object_voxels > 0:
        print(f"    Optional region: removing objects smaller than {config.min_object_voxels} voxels...")
        region_mask = remove_small_objects(
            region_mask,
            max_size=_legacy_threshold_to_max_size(config.min_object_voxels),
        )

    if config.min_hole_voxels > 0:
        print(f"    Optional region: removing holes smaller than {config.min_hole_voxels} voxels...")
        region_mask = remove_small_holes(
            region_mask,
            max_size=_legacy_threshold_to_max_size(config.min_hole_voxels),
        )

    if roi_mask_3d is not None:
        region_mask[~roi_mask_3d] = False

    region_labels = label(region_mask).astype(np.uint32)

    print(f"    Optional region: threshold={threshold}")
    print(f"    Optional region: n_objects={int(region_labels.max())}")

    return OptionalRegionSegmentationResult(
        mask=region_mask,
        labels=region_labels,
        threshold=threshold,
        corrected_image=np.asarray(image_corrected),
    )
