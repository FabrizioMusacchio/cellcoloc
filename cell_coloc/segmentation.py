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
from .schemas import CellposeRefinementRoiCache, OptionalRegionSegmentationResult


def resolve_cellpose_anisotropy(
    model_config: CellposeModelConfig,
    voxel_scale_zyx: tuple[float, float, float],
    do_3d: bool,
) -> float | None:
    """Resolve the Cellpose anisotropy factor for one segmentation run.

    The function only applies anisotropy handling to genuine 3D runs. Users
    can disable the feature with ``False``, enable automatic derivation from
    voxel spacing with ``True``, or provide a numeric factor explicitly.

    The automatic rule follows a practical microscopy heuristic: if the z-step
    is appreciably larger than the in-plane sampling, Cellpose benefits from an
    anisotropy factor of ``z_spacing / mean(xy_spacing)``. If z-spacing is not
    larger than the in-plane spacing, no anisotropy value is forwarded.
    """

    if not do_3d:
        return None

    anisotropy_setting = model_config.anisotropy
    if anisotropy_setting is False:
        return None

    if isinstance(anisotropy_setting, (int, float)) and not isinstance(
        anisotropy_setting,
        bool,
    ):
        anisotropy_value = float(anisotropy_setting)
        if anisotropy_value <= 0:
            raise ValueError(
                "A manually configured Cellpose anisotropy value must be "
                f"greater than 0, got {anisotropy_value}."
            )
        print(f"Using manually configured Cellpose anisotropy: {anisotropy_value:.4f}")
        return anisotropy_value

    if anisotropy_setting is not True:
        raise ValueError(
            "`CellposeModelConfig.anisotropy` must be set to False, True, or "
            f"a positive numeric value, got {anisotropy_setting!r}."
        )

    z_spacing, y_spacing, x_spacing = voxel_scale_zyx
    if z_spacing <= 0 or y_spacing <= 0 or x_spacing <= 0:
        raise ValueError(
            "Voxel spacing values must be strictly positive to derive "
            f"anisotropy automatically, got {voxel_scale_zyx}."
        )

    xy_spacing = (y_spacing + x_spacing) / 2.0
    anisotropy_activation_ratio = 1.25
    if z_spacing <= xy_spacing * anisotropy_activation_ratio:
        print(
            "Skipping Cellpose anisotropy auto-correction because z-spacing "
            f"({z_spacing:.4f}) is not sufficiently larger than mean "
            f"xy-spacing ({xy_spacing:.4f})."
        )
        return None

    anisotropy_value = z_spacing / xy_spacing
    print(
        "Using automatically derived Cellpose anisotropy: "
        f"{anisotropy_value:.4f} (z={z_spacing:.4f}, mean_xy={xy_spacing:.4f})"
    )
    return anisotropy_value


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


def create_cellpose_models_for_channels(
    cell_model_config: CellposeModelConfig,
    marker_model_config: CellposeModelConfig,
    use_gpu: bool,
) -> tuple[models.CellposeModel, models.CellposeModel]:
    """Create Cellpose model instances for the cell and marker channels.

    For Cellpose 4 and newer, the same model instance is reused when both
    channels request the same built-in model or custom model path. For older
    Cellpose versions, the previous behavior is preserved and separate model
    instances are created for each channel configuration.
    """

    cellpose_major = get_cellpose_major_version()
    if cellpose_major is not None and cellpose_major >= 4:
        cell_model_name = cell_model_config.model_name_or_path
        marker_model_name = marker_model_config.model_name_or_path
        if cell_model_name == marker_model_name:
            shared_model = create_cellpose_model(cell_model_name, use_gpu)
            print("Reusing one shared Cellpose model instance for both channels.")
            return shared_model, shared_model

    return (
        create_cellpose_model(cell_model_config.model_name_or_path, use_gpu),
        create_cellpose_model(marker_model_config.model_name_or_path, use_gpu),
    )


def evaluate_cellpose_model(
    model: models.CellposeModel,
    image_zyx: np.ndarray,
    model_config: CellposeModelConfig,
    voxel_scale_zyx: tuple[float, float, float],
) -> tuple[np.ndarray, CellposeRefinementRoiCache | None]:
    """Run Cellpose and return the resulting label image as ``uint32``.

    The function accepts both 3D ``ZYX`` arrays and 2D images represented as a
    singleton-z ``(1, Y, X)`` volume. When ``model_config.do_3d`` is ``None``,
    the function auto-detects the correct Cellpose mode from the z-size.
    """

    do_3d = model_config.do_3d
    if do_3d is None:
        do_3d = image_zyx.shape[0] > 1
    anisotropy = resolve_cellpose_anisotropy(
        model_config=model_config,
        voxel_scale_zyx=voxel_scale_zyx,
        do_3d=do_3d,
    )
    cellpose_major = get_cellpose_major_version()

    if not do_3d and image_zyx.shape[0] != 1:
        raise ValueError(
            "A 2D Cellpose run was requested for an image with more than one z "
            "slice. Please keep automatic 3D detection enabled or project the "
            "image to 2D before segmentation."
        )

    cellpose_input = image_zyx if do_3d else image_zyx[0]
    shape_for_masks = image_zyx.shape if do_3d else (1, image_zyx.shape[1], image_zyx.shape[2])
    eval_kwargs = {
        "do_3D": do_3d,
        "z_axis": model_config.z_axis if do_3d else None,
        "channel_axis": model_config.channel_axis,
    }
    if anisotropy is not None:
        eval_kwargs["anisotropy"] = anisotropy

    if cellpose_major is not None and cellpose_major >= 4:
        eval_kwargs["cellprob_threshold"] = model_config.cellprob_threshold
        eval_kwargs["flow_threshold"] = model_config.flow_threshold
        eval_kwargs["compute_masks"] = False
        if model_config.diameter is not None:
            eval_kwargs["diameter"] = model_config.diameter
    else:
        if model_config.diameter is None:
            raise ValueError(
                "For Cellpose versions below 4, an explicit diameter is "
                "currently required by this pipeline. Please set "
                "`CellposeModelConfig.diameter` in the user script."
            )
        eval_kwargs["diameter"] = model_config.diameter

    masks, flows, _ = model.eval(cellpose_input, **eval_kwargs)

    if cellpose_major is not None and cellpose_major >= 4:
        dP = np.asarray(flows[1])
        cellprob = np.asarray(flows[2])
        if do_3d:
            if dP.ndim != 4:
                raise ValueError(
                    "Cellpose returned unexpected 3D flow dimensions for "
                    f"`dP`: {dP.shape}."
                )
            if cellprob.ndim != 3:
                raise ValueError(
                    "Cellpose returned unexpected 3D cellprob dimensions: "
                    f"{cellprob.shape}."
                )
        else:
            if dP.ndim == 3:
                dP = dP[:, np.newaxis, :, :]
            elif dP.ndim != 4:
                raise ValueError(
                    "Cellpose returned unexpected 2D flow dimensions for "
                    f"`dP`: {dP.shape}."
                )

            if cellprob.ndim == 2:
                cellprob = cellprob[np.newaxis, :, :]
            elif cellprob.ndim != 3:
                raise ValueError(
                    "Cellpose returned unexpected 2D cellprob dimensions: "
                    f"{cellprob.shape}."
                )
        image_scaling = 30.0 / model_config.diameter if model_config.diameter is not None and model_config.diameter > 0 else 1.0
        niter = int(200 / image_scaling)
        masks = model._compute_masks(
            shape_for_masks,
            dP,
            cellprob,
            flow_threshold=model_config.flow_threshold,
            cellprob_threshold=model_config.cellprob_threshold,
            min_size=15,
            max_size_fraction=0.4,
            niter=niter,
            do_3D=do_3d,
            stitch_threshold=0.0,
        )
        refinement_cache = CellposeRefinementRoiCache(
            roi_id=-1,
            y_min=-1,
            y_max=-1,
            x_min=-1,
            x_max=-1,
            roi_mask_crop_2d=np.zeros((1, 1), dtype=bool),
            shape_for_masks=tuple(int(v) for v in shape_for_masks),
            dP=dP,
            cellprob=cellprob,
            do_3d=do_3d,
            niter=niter,
            min_size=15,
            max_size_fraction=0.4,
            flow_threshold=model_config.flow_threshold,
            cellprob_threshold=model_config.cellprob_threshold,
        )
    else:
        refinement_cache = None

    masks_array = np.asarray(masks, dtype=np.uint32)
    if do_3d:
        return masks_array, refinement_cache

    if masks_array.ndim != 2:
        raise ValueError(
            "Cellpose returned an unexpected 2D mask shape: "
            f"{masks_array.shape}."
        )

    return masks_array[np.newaxis, :, :], refinement_cache


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
