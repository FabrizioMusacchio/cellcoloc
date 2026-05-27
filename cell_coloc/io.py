"""Input and output helpers for the cell colocalization pipeline."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

_CACHE_ROOT = Path(tempfile.gettempdir()) / "cell_coloc_runtime_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "numba").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "napari").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(_CACHE_ROOT / "numba"))
os.environ.setdefault("NAPARI_CONFIG", str(_CACHE_ROOT / "napari"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg_cache"))

import omio as om
import pandas as pd
import tifffile

from .config import ChannelConfig
from .schemas import ColocalizationRunResult, LoadedImageChannels, OptionalRegionSegmentationResult, ResultsPaths


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
    """Extract one zero-based channel from an image assumed to be in TZCYX order."""

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
    voxel_scale_zyx: tuple[float, float, float],
    crop_for_testing: tuple[slice, slice, slice] | None = None,
) -> LoadedImageChannels:
    """Load the configured channels from a microscopy dataset.

    The function relies on :mod:`omio` so that future projects are not limited
    to CZI files. The loaded channels are returned as ZYX volumes.
    """

    paths = build_results_paths(source_path)
    image_tzcyx, metadata = om.imread(paths.source_path)
    image_tzcyx = np.asarray(image_tzcyx)

    print(f"Loaded image: {paths.source_path}")
    print(f"Raw image shape (expected TZCYX): {image_tzcyx.shape}")

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
        voxel_scale_zyx=voxel_scale_zyx,
        cell_image=cell_image,
        marker_image=marker_image,
        optional_region_image=optional_region_image,
        raw_shape_tzcyx=tuple(image_tzcyx.shape),
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

    if optional_region_result is not None:
        tifffile.imwrite(paths.optional_region_mask_path, optional_region_result.labels.astype(np.uint32))

    with pd.ExcelWriter(paths.excel_path) as writer:
        run_result.tables.detailed.to_excel(writer, sheet_name="detailed_overlaps", index=False)
        run_result.tables.summary.to_excel(writer, sheet_name="cell_summary", index=False)
        run_result.tables.overview.to_excel(writer, sheet_name="roi_overview", index=False)

    print(f"Saved CSV analysis to:\n{paths.detailed_csv_path}")
    print(f"Saved Excel analysis to:\n{paths.excel_path}")
    print(f"Saved cell masks to:\n{paths.cell_mask_path}")
    print(f"Saved marker masks to:\n{paths.marker_mask_path}")
    print(f"Saved positive cell masks to:\n{paths.positive_cell_mask_path}")

    if optional_region_result is not None:
        print(f"Saved optional region mask to:\n{paths.optional_region_mask_path}")
