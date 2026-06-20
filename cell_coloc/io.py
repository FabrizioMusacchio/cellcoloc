"""Input and output helpers for the cell colocalization pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import omio as om
import pandas as pd
import tifffile

from .config import ChannelConfig
from .schemas import ColocalizationRunResult, LoadedImageChannels, OptionalRegionSegmentationResult, ResultsPaths


def _convert_length_to_microns(value: float, unit: str | None, axis_name: str) -> float:
    """Convert one physical pixel-size value to micrometers.

    Parameters
    ----------
    value:
        Raw physical size from OMIO metadata.
    unit:
        Corresponding physical unit string from OMIO metadata.
    axis_name:
        Human-readable axis name used for error messages.

    Returns
    -------
    float
        Physical size converted to micrometers.
    """

    if value is None:
        raise ValueError(f"Missing physical size value for axis {axis_name}.")
    if float(value) <= 0:
        raise ValueError(f"Physical size for axis {axis_name} must be positive, got {value!r}.")

    normalized_unit = "micron" if unit is None else str(unit).strip().lower()
    unit_to_micron_factor = {
        "micron": 1.0,
        "microns": 1.0,
        "um": 1.0,
        "µm": 1.0,
        "micrometer": 1.0,
        "micrometers": 1.0,
        "micrometre": 1.0,
        "micrometres": 1.0,
        "nm": 1e-3,
        "nanometer": 1e-3,
        "nanometers": 1e-3,
        "nanometre": 1e-3,
        "nanometres": 1e-3,
        "mm": 1e3,
        "millimeter": 1e3,
        "millimeters": 1e3,
        "millimetre": 1e3,
        "millimetres": 1e3,
        "m": 1e6,
        "meter": 1e6,
        "meters": 1e6,
        "metre": 1e6,
        "metres": 1e6,
    }

    if normalized_unit not in unit_to_micron_factor:
        raise ValueError(
            f"Unsupported physical size unit for axis {axis_name}: {unit!r}. "
            "Please provide `voxel_scale_zyx` explicitly for this dataset."
        )

    return float(value) * unit_to_micron_factor[normalized_unit]


def _resolve_voxel_scale_zyx(
    voxel_scale_zyx: tuple[float, float, float] | None,
    metadata,
) -> tuple[float, float, float]:
    """Resolve voxel size in ZYX order from user input or OMIO metadata.

    Resolution order is:

    1. Explicit user-provided ``voxel_scale_zyx``
    2. OMIO metadata entries ``PhysicalSizeZ/Y/X`` and their units
    3. Fallback to ``(1.0, 1.0, 1.0)`` with a warning

    Returns
    -------
    tuple[float, float, float]
        Resolved voxel size in ``(Z, Y, X)`` order expressed in micrometers.
    """

    if voxel_scale_zyx is not None:
        resolved = tuple(float(value) for value in voxel_scale_zyx)
        if len(resolved) != 3 or any(value <= 0 for value in resolved):
            raise ValueError(
                "`voxel_scale_zyx` must be a tuple of three positive values in "
                f"ZYX order, got {voxel_scale_zyx!r}."
            )
        print(f"Using user-provided voxel scale (ZYX, um): {resolved}")
        return resolved

    try:
        resolved = (
            _convert_length_to_microns(metadata["PhysicalSizeZ"], metadata.get("PhysicalSizeZUnit"), "Z"),
            _convert_length_to_microns(metadata["PhysicalSizeY"], metadata.get("PhysicalSizeYUnit"), "Y"),
            _convert_length_to_microns(metadata["PhysicalSizeX"], metadata.get("PhysicalSizeXUnit"), "X"),
        )
        print(f"Using voxel scale from OMIO metadata (ZYX, um): {resolved}")
        return resolved
    except Exception as exc:
        fallback = (1.0, 1.0, 1.0)
        print(
            "Could not resolve voxel scale from user input or OMIO metadata. "
            f"Falling back to {fallback} um in ZYX order. Reason: {exc}"
        )
        return fallback


def build_results_paths(source_path: Path) -> ResultsPaths:
    """Create the standard results paths for one microscopy dataset.

    Parameters
    ----------
    source_path:
        Path to the raw microscopy file.

    Returns
    -------
    ResultsPaths
        Structured output paths inside a ``results`` subdirectory located next
        to the raw input file.
    """

    source_path = Path(source_path).expanduser().resolve()
    results_dir = source_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    stem = source_path.stem
    return ResultsPaths(
        source_path=source_path,
        results_dir=results_dir,
        roi_mask_path=results_dir / f"{stem}_roi_labelmask.tif",
        detailed_csv_path=results_dir / f"{stem}_cell_colocalization.csv",
        excel_path=results_dir / f"{stem}_cell_colocalization.xlsx",
        cell_mask_path=results_dir / f"{stem}_cell_masks.tif",
        marker_mask_path=results_dir / f"{stem}_marker_masks.tif",
        positive_cell_mask_path=results_dir / f"{stem}_positive_cell_masks.tif",
        optional_region_mask_path=results_dir / f"{stem}_region_mask.tif",
    )


def _extract_zyx_channel(image_tzcyx: np.ndarray, channel_index: int) -> np.ndarray:
    """Extract one zero-based channel from a ``TZCYX`` image as a ``ZYX`` volume.

    A singleton time axis is removed automatically. True 2D channels are
    normalized to shape ``(1, Y, X)`` so downstream code can handle 2D and 3D
    data through the same interface.
    """

    if image_tzcyx.ndim != 5:
        raise ValueError(
            "Expected an OMIO image in TZCYX order with five dimensions, "
            f"but received shape {image_tzcyx.shape}."
        )

    if channel_index < 0 or channel_index >= image_tzcyx.shape[2]:
        raise ValueError(
            f"Requested channel {channel_index}, but the C axis has size "
            f"{image_tzcyx.shape[2]}."
        )

    channel = image_tzcyx[:, :, channel_index, :, :]
    channel = np.asarray(channel).squeeze()

    if channel.ndim == 2:
        channel = channel[np.newaxis, :, :]

    if channel.ndim != 3:
        raise ValueError(
            "The extracted channel could not be converted to a ZYX volume. "
            f"Received shape {channel.shape} after squeezing."
        )

    return channel


def load_analysis_images(
    source_path: Path,
    channel_config: ChannelConfig,
    voxel_scale_zyx: tuple[float, float, float] | None,
    crop_for_testing: tuple[slice, slice, slice] | None = None,
    image_loading_mode: str = "memory",
) -> LoadedImageChannels:
    """Load the configured channels from a microscopy dataset.

    The function relies on :mod:`omio` so that future projects are not limited
    to CZI files. The loaded channels are returned as ``ZYX`` volumes. Two
    loading modes are supported:

    - ``"memory"``: eager in-memory loading with ``zarr_store=None``
    - ``"memap"``: disk-backed OMIO/Zarr cache with ``zarr_store="disk"``
      and ``reuse_disk_cache=True``

    Parameters
    ----------
    source_path:
        Input microscopy dataset that OMIO can open.
    channel_config:
        Zero-based channel mapping defining which raw channels correspond to
        the cell, marker, and optional third channel.
    voxel_scale_zyx:
        Optional explicit voxel size in micrometers and ``(Z, Y, X)`` order.
        When this is ``None``, the loader first tries to derive physical pixel
        sizes from OMIO metadata entries such as ``PhysicalSizeZ`` and falls
        back to ``(1.0, 1.0, 1.0)`` with a warning when no usable metadata are
        available.
    crop_for_testing:
        Optional test crop applied after channel extraction in ``(Z, Y, X)``
        order.
    image_loading_mode:
        Raw-image loading strategy. ``"memory"`` materializes the full image
        eagerly, whereas ``"memap"`` keeps OMIO's disk-backed Zarr cache.

    Returns
    -------
    LoadedImageChannels
        Structured bundle containing the extracted analysis channels, resolved
        voxel size, OMIO metadata, and standardized output paths.
    """

    paths = build_results_paths(source_path)
    normalized_loading_mode = image_loading_mode.strip().lower()
    if normalized_loading_mode == "memory":
        image_tzcyx, metadata = om.imread(paths.source_path, zarr_store=None)
        image_tzcyx = np.asarray(image_tzcyx)
    elif normalized_loading_mode == "memap":
        image_tzcyx, metadata = om.imread(
            paths.source_path,
            zarr_store="disk",
            reuse_disk_cache=True,
        )
    else:
        raise ValueError(
            "`image_loading_mode` must be 'memory' or 'memap', got "
            f"{image_loading_mode!r}."
        )

    print(f"Image loading mode: {normalized_loading_mode}")
    raw_z_size = int(image_tzcyx.shape[1])
    is_3d = raw_z_size > 1

    print(f"Loaded image: {paths.source_path}")
    print(f"Raw image shape (expected TZCYX): {image_tzcyx.shape}")
    print(f"Detected dimensionality from Z axis: {'3D' if is_3d else '2D'} (Z={raw_z_size})")

    resolved_voxel_scale_zyx = _resolve_voxel_scale_zyx(voxel_scale_zyx, metadata)

    cell_image = _extract_zyx_channel(image_tzcyx, channel_config.cell_channel)
    marker_image = _extract_zyx_channel(image_tzcyx, channel_config.marker_channel)

    optional_region_image = None
    if channel_config.optional_region_channel is not None:
        optional_region_image = _extract_zyx_channel(
            image_tzcyx,
            channel_config.optional_region_channel,
        )

    if crop_for_testing is not None:
        cell_image = cell_image[crop_for_testing]
        marker_image = marker_image[crop_for_testing]
        if optional_region_image is not None:
            optional_region_image = optional_region_image[crop_for_testing]

    if cell_image.shape != marker_image.shape:
        raise ValueError(
            "Cell and marker images must have identical shapes, but received "
            f"{cell_image.shape} and {marker_image.shape}."
        )

    if optional_region_image is not None and optional_region_image.shape != cell_image.shape:
        raise ValueError(
            "The optional third channel must match the primary analysis shape, "
            f"but received {optional_region_image.shape} and {cell_image.shape}."
        )

    print(f"Analysis volume shape (ZYX): {cell_image.shape}")

    return LoadedImageChannels(
        source_path=paths.source_path,
        paths=paths,
        voxel_scale_zyx=resolved_voxel_scale_zyx,
        cell_image=cell_image,
        marker_image=marker_image,
        optional_region_image=optional_region_image,
        raw_shape_tzcyx=tuple(image_tzcyx.shape),
        raw_z_size=raw_z_size,
        is_3d=is_3d,
        metadata=metadata,
    )


def save_roi_labels(path: Path, roi_labels_2d: np.ndarray) -> None:
    """Persist a 2D ROI label mask as ``uint16`` TIFF."""

    tifffile.imwrite(Path(path), roi_labels_2d.astype(np.uint16))
    print(f"Saved ROI label mask to:\n{Path(path)}")


def load_roi_labels(path: Path) -> np.ndarray:
    """Load a previously saved 2D ROI label mask."""

    roi_labels = tifffile.imread(Path(path)).astype(np.uint16)
    print(f"Loaded ROI label mask from:\n{Path(path)}")
    return roi_labels


def try_load_roi_labels(path: Path) -> np.ndarray | None:
    """Load a saved 2D ROI label mask when it exists.

    Parameters
    ----------
    path:
        Expected path of the ROI label TIFF.

    Returns
    -------
    np.ndarray | None
        The loaded ROI label mask when the file exists, otherwise ``None``.
        This helper is useful for interactive scripts that should prefer a
        previously drawn ROI mask but fall back to manual drawing when no saved
        mask is available yet.
    """

    path = Path(path)
    if not path.exists():
        print(f"No existing ROI label mask found at:\n{path}")
        return None

    return load_roi_labels(path)


def export_analysis_outputs(
    run_result: ColocalizationRunResult,
    paths: ResultsPaths,
    optional_region_result: OptionalRegionSegmentationResult | None = None,
) -> None:
    """Write all standard tables and masks for one completed analysis run.

    Parameters
    ----------
    run_result:
        Completed Cellpose colocalization result bundle.
    paths:
        Standardized output paths created for the current dataset.
    optional_region_result:
        Optional third-channel segmentation result. When provided, the labeled
        mask is exported alongside the main colocalization outputs.
    """

    run_result.tables.detailed.to_csv(paths.detailed_csv_path, index=False)

    tifffile.imwrite(paths.cell_mask_path, run_result.cell_masks.astype(np.uint32))
    tifffile.imwrite(paths.marker_mask_path, run_result.marker_masks.astype(np.uint32))
    tifffile.imwrite(paths.positive_cell_mask_path, run_result.positive_cell_masks.astype(np.uint32))

    optional_region_labels_to_export = None
    if run_result.optional_region_masks is not None:
        optional_region_labels_to_export = run_result.optional_region_masks
    elif optional_region_result is not None:
        optional_region_labels_to_export = optional_region_result.labels

    if optional_region_labels_to_export is not None:
        tifffile.imwrite(paths.optional_region_mask_path, optional_region_labels_to_export.astype(np.uint32))

    with pd.ExcelWriter(paths.excel_path) as writer:
        run_result.tables.detailed.to_excel(writer, sheet_name="detailed_overlaps", index=False)
        run_result.tables.summary.to_excel(writer, sheet_name="cell_summary", index=False)
        run_result.tables.overview.to_excel(writer, sheet_name="roi_overview", index=False)

    print(f"Saved CSV analysis to:\n{paths.detailed_csv_path}")
    print(f"Saved Excel analysis to:\n{paths.excel_path}")
    print(f"Saved cell masks to:\n{paths.cell_mask_path}")
    print(f"Saved marker masks to:\n{paths.marker_mask_path}")
    print(f"Saved positive cell masks to:\n{paths.positive_cell_mask_path}")

    if optional_region_labels_to_export is not None:
        print(f"Saved optional region mask to:\n{paths.optional_region_mask_path}")
