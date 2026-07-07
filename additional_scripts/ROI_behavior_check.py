"""Debugging script for ROI-size sensitivity in the z-projected microglia demo.

This script reuses the same core settings as the public
``microglia_3D_three_channel_zproject_user_script.py`` workflow, but replaces
interactive ROI drawing with a deterministic loop over automatically generated
ROI pairs. The two ROIs stay concentric across iterations and grow gradually,
which makes it easier to inspect whether the detected cell masks change as a
function of ROI size.

Each loop iteration writes its outputs into a dedicated subfolder so no files
are overwritten:

- ROI label mask
- exported CellColoc CSV/XLSX outputs
- segmentation masks
- a simple PNG preview for quick visual comparison

author: Fabrizio Musacchio
date:   July 2026
"""

# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd

from cellcoloc import (
    CellposeModelConfig,
    ChannelConfig,
    ColocalizationConfig,
    DisplayNames,
    ResultsPaths,
    RuntimeConfig,
    export_analysis_outputs,
    load_analysis_images,
    prepare_loaded_images_for_analysis,
    run_roi_cellpose_colocalization,
    save_roi_labels,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# %% PROJECT SETTINGS
DATA_DIR = PROJECT_ROOT / "example_data" / "microglia_3D"
DATA_PATHS = sorted(DATA_DIR.glob("*"))
allowed_extensions = [".czi", ".tif", ".tiff", ".ome.tif", ".ome.tiff"]
DATA_PATHS = [p for p in DATA_PATHS if p.suffix.lower() in allowed_extensions]
DATA_PATHS = [p for p in DATA_PATHS if not p.name.startswith(".")]

CHANNEL_CONFIG = ChannelConfig(
    cell_channel=0,
    marker_channel=1,
    optional_region_channel=2,
)

DISPLAY_NAMES = DisplayNames(
    cell="Cx3cr1-tdTomato",
    marker="Iba1",
    optional_region="DAPI",
    positive_cells="tdTomato + Iba1 positive masks",
)

VOXEL_SCALE_ZYX = None

CELL_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",
    segmentation_method="cellpose",
    diameter=None,
    z_crop=None,
    z_projection="max",
    anisotropy=True,
    flow3d_smooth=0,
    prefilter=None,
    prefilter_sigma_xy=0.0,
    prefilter_sigma_z=0.0,
    prefilter_median_size_xy=3,
    prefilter_median_size_z=3,
    postfilters=None,
    min_intensity_measure="mean",
    min_intensity_threshold=None,
    local_contrast_k=1.0,
    local_contrast_shell_inner_radius=1,
    local_contrast_shell_outer_radius=4,
    bright_pixel_measure="count",
    bright_pixel_threshold=None,
    bright_pixel_min_count=None,
    bright_pixel_min_fraction=None,
    cellprob_threshold=1.5,
    flow_threshold=0.4,
)

MARKER_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",
    segmentation_method="otsu",
    diameter=None,
    anisotropy=True,
    flow3d_smooth=0,
    prefilter=None,
    prefilter_sigma_xy=0.0,
    prefilter_sigma_z=0.0,
    prefilter_median_size_xy=3,
    prefilter_median_size_z=3,
    postfilters=None,
    min_intensity_measure="mean",
    min_intensity_threshold=None,
    local_contrast_k=1.0,
    local_contrast_shell_inner_radius=1,
    local_contrast_shell_outer_radius=4,
    bright_pixel_measure="count",
    bright_pixel_threshold=None,
    bright_pixel_min_count=None,
    bright_pixel_min_fraction=None,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
)

OPTIONAL_REGION_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="cpsam",
    segmentation_method="cellpose",
    diameter=None,
    anisotropy=True,
    flow3d_smooth=0,
    prefilter=None,
    prefilter_sigma_xy=0.0,
    prefilter_sigma_z=0.0,
    prefilter_median_size_xy=3,
    prefilter_median_size_z=3,
    postfilters=None,
    min_intensity_measure="mean",
    min_intensity_threshold=None,
    local_contrast_k=1.0,
    local_contrast_shell_inner_radius=1,
    local_contrast_shell_outer_radius=4,
    bright_pixel_measure="count",
    bright_pixel_threshold=None,
    bright_pixel_min_count=None,
    bright_pixel_min_fraction=None,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
)

COLOCALIZATION_CONFIG = ColocalizationConfig(
    min_cell_voxels=50,
    overlap_fraction_threshold=0.02,
    min_overlap_voxels=10,
    evaluate_optional_region_cell_positivity=True,
)

RUNTIME_CONFIG = RuntimeConfig(
    draw_rois=False,
    process_rois=True,
    open_results=False,
    use_gpu=True,
    crop_for_testing=None,
    image_loading_mode="memap",
)

ROI_SCALE_FACTORS = [0.70, 0.90, 1.10, 1.35, 1.50, 2.0, 3.0]
SAVE_PREVIEW_PNGS = True

print("Detected input files:")
for data_path in DATA_PATHS:
    print(f" - {data_path.name}")

SELECTED_FILE_NAME = DATA_PATHS[0].name
DATA_PATH = DATA_DIR / SELECTED_FILE_NAME
print(f"Selected file for analysis:\n{DATA_PATH}")


# %% LOAD THE ANALYSIS CHANNELS
loaded_images = load_analysis_images(
    source_path=DATA_PATH,
    channel_config=CHANNEL_CONFIG,
    voxel_scale_zyx=VOXEL_SCALE_ZYX,
    crop_for_testing=RUNTIME_CONFIG.crop_for_testing,
    image_loading_mode=RUNTIME_CONFIG.image_loading_mode,
)
loaded_images = prepare_loaded_images_for_analysis(
    loaded_images,
    CELL_MODEL_CONFIG,
    MARKER_MODEL_CONFIG,
    OPTIONAL_REGION_MODEL_CONFIG,
)
print(
    "Prepared analysis view: "
    f"shape={loaded_images.cell_image.shape}, "
    f"is_3d={loaded_images.is_3d}, "
    f"z_projection={loaded_images.z_projection_method!r}, "
    f"analysis_z_bounds={loaded_images.analysis_z_bounds}"
)


# %% DEBUG LOOP OVER GROWING CONCENTRIC ROI PAIRS
@dataclass(slots=True)
class RoiSeed:
    center_yx: tuple[float, float]
    radii_yx: tuple[float, float]


def _ellipse_mask(
    shape_yx: tuple[int, int],
    center_yx: tuple[float, float],
    radii_yx: tuple[float, float],
) -> np.ndarray:
    """Return a filled ellipse mask for one ROI."""

    yy, xx = np.ogrid[: shape_yx[0], : shape_yx[1]]
    center_y, center_x = center_yx
    radius_y, radius_x = radii_yx
    if radius_y <= 0 or radius_x <= 0:
        raise ValueError(f"ROI radii must be positive, got {radii_yx}.")
    normalized = ((yy - center_y) / radius_y) ** 2 + ((xx - center_x) / radius_x) ** 2
    return normalized <= 1.0


def _build_iteration_roi_labels(
    shape_yx: tuple[int, int],
    roi_seeds: list[RoiSeed],
    scale_factor: float,
) -> np.ndarray:
    """Generate one two-ROI label image for a given scale factor."""

    roi_labels = np.zeros(shape_yx, dtype=np.uint16)
    for roi_index, seed in enumerate(roi_seeds, start=1):
        scaled_radii = (
            seed.radii_yx[0] * scale_factor,
            seed.radii_yx[1] * scale_factor,
        )
        roi_mask = _ellipse_mask(shape_yx, seed.center_yx, scaled_radii)
        roi_labels[roi_mask] = np.uint16(roi_index)
    return roi_labels


def _make_iteration_paths(
    source_path: Path,
    debug_root_dir: Path,
    iteration_name: str,
) -> ResultsPaths:
    """Build dedicated output paths for one loop iteration."""

    iteration_dir = debug_root_dir / iteration_name
    iteration_dir.mkdir(parents=True, exist_ok=True)
    stem = source_path.stem
    prefix = f"{stem}_{iteration_name}"
    return ResultsPaths(
        source_path=source_path,
        results_dir=iteration_dir,
        roi_mask_path=iteration_dir / f"{prefix}_roi_labelmask.tif",
        detailed_csv_path=iteration_dir / f"{prefix}_cell_colocalization.csv",
        excel_path=iteration_dir / f"{prefix}_cell_colocalization.xlsx",
        cell_mask_path=iteration_dir / f"{prefix}_cell_masks.tif",
        marker_mask_path=iteration_dir / f"{prefix}_marker_masks.tif",
        positive_cell_mask_path=iteration_dir / f"{prefix}_positive_cell_masks.tif",
        optional_region_mask_path=iteration_dir / f"{prefix}_region_mask.tif",
    )


def _save_iteration_preview(
    preview_path: Path,
    loaded_images,
    roi_labels_2d: np.ndarray,
    run_result,
) -> None:
    """Save a simple visual overview for one loop iteration."""

    cell_image_yx = np.asarray(loaded_images.cell_image[0], dtype=np.float32)
    cell_mask_projection = np.asarray(run_result.cell_masks.max(axis=0), dtype=np.uint32)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes[0].imshow(cell_image_yx, cmap="magma")
    axes[0].imshow(np.ma.masked_where(roi_labels_2d == 0, roi_labels_2d), cmap="autumn", alpha=0.28)
    axes[0].set_title("Cell channel with ROIs")
    axes[0].axis("off")

    axes[1].imshow(np.ma.masked_where(roi_labels_2d == 0, roi_labels_2d), cmap="autumn", alpha=0.28)
    axes[1].imshow(cell_mask_projection, cmap="nipy_spectral", alpha=(cell_mask_projection > 0) * 0.95)
    axes[1].set_title("Cell masks")
    axes[1].axis("off")

    fig.savefig(preview_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


image_shape_yx = tuple(int(v) for v in loaded_images.cell_image.shape[1:])
image_height, image_width = image_shape_yx

roi_seeds = [
    RoiSeed(
        center_yx=(image_height * 0.35, image_width * 0.28),
        radii_yx=(image_height * 0.10, image_width * 0.08),
    ),
    RoiSeed(
        center_yx=(image_height * 0.63, image_width * 0.72),
        radii_yx=(image_height * 0.11, image_width * 0.09),
    ),
]

debug_root_dir = loaded_images.paths.results_dir / f"{loaded_images.source_path.stem}_roi_growth_debug"
debug_root_dir.mkdir(parents=True, exist_ok=True)
print(f"Writing ROI-growth debug outputs to:\n{debug_root_dir}")

iteration_summary_rows: list[dict[str, int | float | str]] = []
iteration_layers: list[dict[str, object]] = []

for iteration_index, scale_factor in enumerate(ROI_SCALE_FACTORS, start=1):
    iteration_name = f"loop_{iteration_index:02d}_scale_{scale_factor:.2f}".replace(".", "p")
    print("\n" + "=" * 80)
    print(f"{iteration_name}: generating concentric ROI pair with scale factor {scale_factor:.2f}")

    roi_labels_2d = _build_iteration_roi_labels(
        shape_yx=image_shape_yx,
        roi_seeds=roi_seeds,
        scale_factor=scale_factor,
    )
    iteration_paths = _make_iteration_paths(
        source_path=loaded_images.source_path,
        debug_root_dir=debug_root_dir,
        iteration_name=iteration_name,
    )

    # Save the generated ROI layer for this iteration.
    save_roi_labels(iteration_paths.roi_mask_path, roi_labels_2d)

    # Run the standard ROI-wise three-channel analysis.
    run_result = run_roi_cellpose_colocalization(
        loaded_images=loaded_images,
        roi_labels_2d=roi_labels_2d,
        cell_model_config=CELL_MODEL_CONFIG,
        marker_model_config=MARKER_MODEL_CONFIG,
        optional_region_model_config=OPTIONAL_REGION_MODEL_CONFIG,
        colocalization_config=COLOCALIZATION_CONFIG,
        runtime_config=RUNTIME_CONFIG,
    )

    # Export normal CellColoc outputs into the dedicated loop folder.
    export_analysis_outputs(
        run_result=run_result,
        paths=iteration_paths,
    )

    if SAVE_PREVIEW_PNGS:
        _save_iteration_preview(
            preview_path=iteration_paths.results_dir / f"{loaded_images.source_path.stem}_{iteration_name}_preview.png",
            loaded_images=loaded_images,
            roi_labels_2d=roi_labels_2d,
            run_result=run_result,
        )

    iteration_layers.append(
        {
            "iteration_name": iteration_name,
            "roi_labels": roi_labels_2d.copy(),
            "cell_masks": np.asarray(run_result.cell_masks, dtype=np.uint32).copy(),
        }
    )

    overview_table = run_result.tables.overview.copy()
    for _, overview_row in overview_table.iterrows():
        iteration_summary_rows.append(
            {
                "iteration": iteration_name,
                "scale_factor": scale_factor,
                "roi_id": int(overview_row["roi_id"]),
                "drawn_roi_area_px": int(overview_row["drawn_roi_area_px"]),
                "n_cells": int(overview_row["n_cells"]),
                "n_marker_positive_cells": int(overview_row["n_marker_positive_cells"]),
                "n_marker_objects": int(overview_row["n_marker_objects"]),
                "cell_occupancy_area_px_2d_projection": int(
                    overview_row["cell_occupancy_area_px_2d_projection"]
                ),
            }
        )

    total_cells = int(overview_table["n_cells"].sum()) if not overview_table.empty else 0
    print(
        f"Finished {iteration_name}: "
        f"{len(overview_table)} ROIs, total detected cells={total_cells}"
    )

iteration_summary = pd.DataFrame(iteration_summary_rows).sort_values(
    by=["scale_factor", "roi_id"]
)
iteration_summary_path = debug_root_dir / f"{loaded_images.source_path.stem}_roi_growth_debug_summary.csv"
iteration_summary.to_csv(iteration_summary_path, index=False)
print("\nSaved loop summary to:")
print(iteration_summary_path)


# %% OPEN A NAPARI VIEWER WITH ALL LOOP ROIS AND CELL MASKS
debug_viewer = napari.Viewer()
debug_viewer.add_image(
    loaded_images.cell_image,
    name=f"{DISPLAY_NAMES.cell} original",
    scale=loaded_images.voxel_scale_zyx,
    blending="additive",
    colormap="magenta",
)

for loop_index, layer_data in enumerate(iteration_layers, start=1):
    roi_labels = np.asarray(layer_data["roi_labels"], dtype=np.uint16)
    roi_labels_3d = np.repeat(roi_labels[np.newaxis, :, :], loaded_images.cell_image.shape[0], axis=0)
    cell_masks = np.asarray(layer_data["cell_masks"], dtype=np.uint32)
    layer_suffix = str(layer_data["iteration_name"])

    roi_layer = debug_viewer.add_labels(
        roi_labels_3d,
        name=f"ROIs {layer_suffix}",
        scale=loaded_images.voxel_scale_zyx,
        blending="additive",
    )
    roi_layer.opacity = 0.35
    roi_layer.visible = loop_index == len(iteration_layers)

    cell_mask_layer = debug_viewer.add_labels(
        cell_masks,
        name=f"Cell masks {layer_suffix}",
        scale=loaded_images.voxel_scale_zyx,
        blending="additive",
    )
    cell_mask_layer.opacity = 0.85
    cell_mask_layer.visible = loop_index == len(iteration_layers)

print("\nOpened a napari viewer with the original cell channel plus ROI and cell-mask layers for all loops.")
print("Toggle individual loop layers on and off in napari to compare iterations.")
napari.run()

# %% END
