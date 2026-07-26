"""Batch-analyze the microglia_3D_full CA1 and CTX datasets for the preprint.

Run from the project root in a terminal with:

    conda run -n cellcoloc python additional_scripts/run_microglia_3d_full_batch.py

Example with overwrite enabled:

    conda run -n cellcoloc python additional_scripts/run_microglia_3d_full_batch.py --overwrite

Example with GPU enabled:

    conda run -n cellcoloc python additional_scripts/run_microglia_3d_full_batch.py --use-gpu --overwrite
"""
# %% IMPORTS
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Avoid editable-environment napari/numba cache issues in non-interactive batch runs.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from cellcoloc import (
    CellposeModelConfig,
    ChannelConfig,
    ColocalizationConfig,
    DisplayNames,
    RuntimeConfig,
    create_full_image_roi_labels,
    export_analysis_outputs,
    load_analysis_images,
    prepare_loaded_images_for_analysis,
    run_roi_cellpose_colocalization,
)

# %% CONFIGURATION
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "example_data" / "microglia_3D_full"

CHANNEL_CONFIG = ChannelConfig(
    cell_channel=0,
    marker_channel=1,
    optional_region_channel=None,
)

DISPLAY_NAMES = DisplayNames(
    cell="Cx3cr1-tdTomato",
    marker="Iba1",
    optional_region="DAPI",
    positive_cells="tdTomato cells positive for Iba1",
)

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

OPTIONAL_REGION_MODEL_CONFIG = None

COLOCALIZATION_CONFIG = ColocalizationConfig(
    min_cell_voxels=50,
    overlap_fraction_threshold=0.02,
    min_overlap_voxels=10,
    evaluate_optional_region_cell_positivity=False,
)

RUNTIME_CONFIG = RuntimeConfig(
    draw_rois=False,
    process_rois=True,
    open_results=False,
    use_gpu=False,
    crop_for_testing=None,
    image_loading_mode="memap",
)

CELL_MODEL_CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    "250703_ID22464_ctx_tdTom_DAPI_IBA1_20x_1.czi": {
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.35,
        "prefilter": ["laplacian_of_gaussian"],
        "prefilter_sigma_xy": 0.5,
    },
    "250703_ID22463_CA1_tdTom_DAPI_IBA1_20x_1.czi": {
        "cellprob_threshold": -0.8,
        "flow_threshold": 0.9,
        "prefilter": ["gaussian"],
        "prefilter_sigma_xy": 0.5,
    },
    "250703_ID22463_CA1_tdTom_DAPI_IBA1_20x_2.czi": {
        "cellprob_threshold": -0.8,
        "flow_threshold": 0.9,
        "prefilter": ["gaussian"],
        "prefilter_sigma_xy": 0.5,
    },
    "250703_ID22461_CA1_tdTom_DAPI_IBA1_20x_1.czi": {
        "cellprob_threshold": -0.8,
        "flow_threshold": 0.9,
        "prefilter": ["gaussian"],
        "prefilter_sigma_xy": 0.5,
    },
    "250703_ID22462_CA1_tdTom_DAPI_IBA1_20x_1.czi": {
        "cellprob_threshold": -0.8,
        "flow_threshold": 0.9,
        "prefilter": ["gaussian"],
        "prefilter_sigma_xy": 0.5,
    },
    "250703_ID22462_CA1_tdTom_DAPI_IBA1_20x_2.czi": {
        "cellprob_threshold": -0.8,
        "flow_threshold": 0.9,
        "prefilter": ["gaussian"],
        "prefilter_sigma_xy": 0.5,
    },
    "250703_ID22464_CA1_tdTom_DAPI_IBA1_20x_1.czi": {
        "cellprob_threshold": -0.5,
        "flow_threshold": 0.6,
        "prefilter": ["gaussian"],
        "prefilter_sigma_xy": 0.5,
    },
    "250703_ID22485_CA1_tdTom_DAPI_IBA1_20x_1.czi": {
        "cellprob_threshold": -0.8,
        "flow_threshold": 0.9,
        "prefilter": ["gaussian"],
        "prefilter_sigma_xy": 0.5,
    },
    "250703_ID22486_CA1_tdTom_DAPI_IBA1_20x_1.czi": {
        "cellprob_threshold": -0.8,
        "flow_threshold": 0.9,
        "prefilter": ["gaussian"],
        "prefilter_sigma_xy": 0.5,
    },
    "250703_ID22487_CA1_tdTom_DAPI_IBA1_20x_1.czi": {
        "cellprob_threshold": -0.8,
        "flow_threshold": 0.9,
        "prefilter": ["gaussian"],
        "prefilter_sigma_xy": 0.5,
    },
    "250703_ID22488_CA1_tdTom_DAPI_IBA1_20x_1.czi": {
        "cellprob_threshold": -0.8,
        "flow_threshold": 0.9,
        "prefilter": ["gaussian"],
        "prefilter_sigma_xy": 0.5,
    },
}

# %% FUNCTIONS
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=("CA1", "CTX"),
        help="Subset of regions to process, e.g. CA1 CTX.",
    )
    parser.add_argument(
        "--limit-per-region",
        type=int,
        default=None,
        help="Optional maximum number of stacks per region.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even when sidecar CSV files already exist.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DATA_DIR / "microglia_3d_full_batch_manifest.csv",
        help="Output CSV path for the per-stack manifest.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Optional zero-based region-local start index for sharded batch runs.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Optional region-local stride for sharded batch runs.",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        help="Enable GPU execution for Cellpose if available.",
    )
    parser.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Force CPU execution even if a GPU is available.",
    )
    parser.set_defaults(use_gpu=RUNTIME_CONFIG.use_gpu)
    return parser.parse_args()

def normalize_for_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image.max(axis=0)
    low, high = np.percentile(image, [1.0, 99.7])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    return image

def save_preview_panels(stack_path: Path, loaded_images, run_result) -> None:
    results_dir = stack_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = stack_path.stem

    cell = normalize_for_display(loaded_images.cell_image)
    marker = normalize_for_display(loaded_images.marker_image)
    if loaded_images.optional_region_image is not None:
        optional = normalize_for_display(loaded_images.optional_region_image)
    else:
        optional = np.zeros_like(cell)

    composite = np.zeros((*cell.shape, 3), dtype=np.float32)
    composite[..., 0] = np.clip(cell + optional * 0.7, 0.0, 1.0)
    composite[..., 1] = np.clip(marker + optional * 0.7, 0.0, 1.0)
    composite[..., 2] = np.clip(marker, 0.0, 1.0)

    positive_mask = (run_result.positive_cell_masks > 0).max(axis=0)
    overlay = composite.copy()
    overlay[..., 1] = np.maximum(overlay[..., 1], positive_mask.astype(np.float32) * 0.95)

    panel_map = {
        f"{stem}_preview_cell.png": cell,
        f"{stem}_preview_marker.png": marker,
        f"{stem}_preview_optional.png": optional,
        f"{stem}_preview_composite.png": composite,
        f"{stem}_preview_positive_overlay.png": overlay,
    }
    for filename, image in panel_map.items():
        plt.imsave(results_dir / filename, image, cmap="gray" if image.ndim == 2 else None)

def resolve_cell_model_config_for_stack(stack_path: Path) -> CellposeModelConfig:
    overrides = CELL_MODEL_CONFIG_OVERRIDES.get(stack_path.name)
    if overrides is None:
        return CELL_MODEL_CONFIG
    return replace(CELL_MODEL_CONFIG, **overrides)


def write_table_sidecars(
    stack_path: Path,
    run_result,
    *,
    cell_model_config: CellposeModelConfig,
) -> None:
    stem = stack_path.stem
    results_dir = stack_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    table_map = {
        f"{stem}_cell_summary.csv": run_result.tables.summary,
        f"{stem}_roi_overview.csv": run_result.tables.overview,
        f"{stem}_marker_properties.csv": run_result.tables.marker_properties,
        f"{stem}_3rd_channel_properties.csv": run_result.tables.third_channel_properties,
        f"{stem}_roi_cell_summary.csv": run_result.tables.roi_cell_summary,
        f"{stem}_roi_marker_summary.csv": run_result.tables.roi_marker_summary,
        f"{stem}_roi_3rd_channel_summary.csv": run_result.tables.roi_third_channel_summary,
    }
    for filename, table in table_map.items():
        if table is None:
            continue
        table.to_csv(results_dir / filename, index=False)

    config_payload = {
        "channel_config": asdict(CHANNEL_CONFIG),
        "cell_model_config": asdict(cell_model_config),
        "marker_model_config": asdict(MARKER_MODEL_CONFIG),
        "optional_region_model_config": (
            asdict(OPTIONAL_REGION_MODEL_CONFIG)
            if OPTIONAL_REGION_MODEL_CONFIG is not None
            else None
        ),
        "colocalization_config": asdict(COLOCALIZATION_CONFIG),
        "runtime_config": asdict(RUNTIME_CONFIG),
    }
    (results_dir / f"{stem}_analysis_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

def collect_region_paths(
    limit_per_region: int | None,
    regions: tuple[str, ...],
    start_index: int,
    stride: int,
) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for region in regions:
        region_paths = sorted((DATA_DIR / region).glob("*.czi"))
        if limit_per_region is not None:
            region_paths = region_paths[:limit_per_region]
        region_paths = region_paths[start_index::stride]
        pairs.extend((region, path) for path in region_paths)
    return pairs


def build_manifest_row(region: str, stack_path: Path, results_dir: Path) -> dict[str, object]:
    overview = pd.read_csv(results_dir / f"{stack_path.stem}_roi_overview.csv")
    summary = pd.read_csv(results_dir / f"{stack_path.stem}_cell_summary.csv")
    n_cells = int(overview["n_cells"].sum())
    n_iba1_positive = int(overview["n_marker_positive_cells"].sum())
    marker_fraction = float(n_iba1_positive / n_cells) if n_cells > 0 else np.nan
    mean_cell_area = float(summary["cell_area_px_2d"].mean()) if "cell_area_px_2d" in summary else np.nan
    mean_cell_roundness = (
        float(summary["cell_roundness_2d"].mean()) if "cell_roundness_2d" in summary else np.nan
    )
    return {
        "region": region,
        "stack_name": stack_path.name,
        "source_path": str(stack_path),
        "results_dir": str(results_dir),
        "n_cells": n_cells,
        "n_iba1_positive_cells": n_iba1_positive,
        "iba1_positive_fraction": marker_fraction,
        "mean_cell_area_px_2d": mean_cell_area,
        "mean_cell_roundness_2d": mean_cell_roundness,
    }

# %% MAIN FUNCTION
def main() -> None:
    args = parse_args()
    runtime_config = replace(RUNTIME_CONFIG, use_gpu=args.use_gpu)
    requested_regions = tuple(dict.fromkeys(args.regions))
    region_paths = collect_region_paths(
        args.limit_per_region,
        requested_regions,
        args.start_index,
        args.stride,
    )
    if not region_paths:
        raise FileNotFoundError(f"No matching CZI files found in {DATA_DIR} for regions {requested_regions}.")

    manifest_rows: list[dict[str, object]] = []
    for region, stack_path in region_paths:
        results_dir = stack_path.parent / "results"
        overview_sidecar = results_dir / f"{stack_path.stem}_roi_overview.csv"
        if overview_sidecar.exists() and not args.overwrite:
            print(f"Using existing analysis: {stack_path.name}")
            manifest_rows.append(build_manifest_row(region=region, stack_path=stack_path, results_dir=results_dir))
            continue

        print(f"Starting {region} / {stack_path.name}")
        cell_model_config = resolve_cell_model_config_for_stack(stack_path)
        if cell_model_config != CELL_MODEL_CONFIG:
            print(f"Applying stack-specific cell-model override for {stack_path.name}")
        loaded_images = load_analysis_images(
            source_path=stack_path,
            channel_config=CHANNEL_CONFIG,
            voxel_scale_zyx=None,
            crop_for_testing=None,
            image_loading_mode=runtime_config.image_loading_mode,
        )
        print(
            f"Loaded {stack_path.name}: "
            f"cell_shape={loaded_images.cell_image.shape}, "
            f"is_3d={loaded_images.is_3d}"
        )
        loaded_images = prepare_loaded_images_for_analysis(
            loaded_images,
            cell_model_config,
            MARKER_MODEL_CONFIG,
            OPTIONAL_REGION_MODEL_CONFIG,
        )
        print(
            f"Prepared {stack_path.name}: "
            f"analysis_shape={loaded_images.cell_image.shape}, "
            f"z_projection={loaded_images.z_projection_method!r}, "
            f"analysis_z_bounds={loaded_images.analysis_z_bounds}"
        )

        roi_labels_2d = create_full_image_roi_labels(loaded_images.cell_image.shape[1:])
        print(f"Running segmentation and overlap analysis for {stack_path.name}...")
        run_result = run_roi_cellpose_colocalization(
            loaded_images=loaded_images,
            roi_labels_2d=roi_labels_2d,
            cell_model_config=cell_model_config,
            marker_model_config=MARKER_MODEL_CONFIG,
            colocalization_config=COLOCALIZATION_CONFIG,
            runtime_config=runtime_config,
            optional_region_model_config=OPTIONAL_REGION_MODEL_CONFIG,
            optional_region_result=None,
        )
        print(f"Finished core analysis for {stack_path.name}. Exporting outputs...")
        export_analysis_outputs(
            run_result=run_result,
            paths=loaded_images.paths,
            optional_region_result=None,
        )
        write_table_sidecars(
            stack_path=stack_path,
            run_result=run_result,
            cell_model_config=cell_model_config,
        )
        save_preview_panels(stack_path=stack_path, loaded_images=loaded_images, run_result=run_result)

        manifest_row = build_manifest_row(region=region, stack_path=stack_path, results_dir=results_dir)
        manifest_rows.append(manifest_row)
        print(
            f"Processed {region} / {stack_path.name}: "
            f"n_cells={manifest_row['n_cells']}, "
            f"n_iba1_positive={manifest_row['n_iba1_positive_cells']}, "
            f"fraction={manifest_row['iba1_positive_fraction']:.3f}"
        )

    manifest = pd.DataFrame(manifest_rows)
    if not manifest.empty:
        args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.sort_values(by=["region", "stack_name"]).to_csv(args.manifest_path, index=False)
        print(f"Saved batch manifest to:\n{args.manifest_path}")

# %% MAIN ENTRY
if __name__ == "__main__":
    main()
# %% END
