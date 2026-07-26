"""Run CellColoc on the synthetic benchmark stacks used in the preprint.

Run from the project root in a terminal with:

    conda run -n cellcoloc python additional_scripts/run_synthetic_benchmark_cellcoloc.py

Example with overwrite enabled:

    conda run -n cellcoloc python additional_scripts/run_synthetic_benchmark_cellcoloc.py --overwrite
"""
# %% IMPORTS
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

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
    run_roi_cellpose_colocalization)
# %% CONFIGURATION
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "example_data" / "synthetic_benchmark_data"

CHANNEL_CONFIG = ChannelConfig(
    cell_channel=0,
    marker_channel=1,
    optional_region_channel=2,
)

DISPLAY_NAMES = DisplayNames(
    cell="Synthetic cells",
    marker="Synthetic marker",
    optional_region="Synthetic region",
    positive_cells="Channel-0 cells positive for channel 1",
)

CELL_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="unused_threshold_backend",
    segmentation_method="otsu",
    do_3d=False,
    prefilter="gaussian",
    prefilter_sigma_xy=1.0,
    threshold_background_sigma=None,
    threshold_min_object_voxels=70,
    threshold_min_hole_voxels=24,
    threshold_apply_closing=True,
)

MARKER_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="unused_threshold_backend",
    segmentation_method="otsu",
    do_3d=False,
    prefilter="gaussian",
    prefilter_sigma_xy=1.0,
    threshold_background_sigma=None,
    threshold_min_object_voxels=55,
    threshold_min_hole_voxels=20,
    threshold_apply_closing=True,
)

OPTIONAL_REGION_MODEL_CONFIG = CellposeModelConfig(
    model_name_or_path="unused_threshold_backend",
    segmentation_method="li",
    do_3d=False,
    prefilter="gaussian",
    prefilter_sigma_xy=1.6,
    threshold_background_sigma=None,
    threshold_min_object_voxels=900,
    threshold_min_hole_voxels=250,
    threshold_apply_closing=True,
)

COLOCALIZATION_CONFIG = ColocalizationConfig(
    min_cell_voxels=70,
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
    image_loading_mode="memory",
)
# %% FUNCTIONS
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of stacks to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even when sidecar CSV files already exist.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Synthetic benchmark directory to process (default: {DATA_DIR}).",
    )
    return parser.parse_args()

def write_table_sidecars(stack_path: Path, run_result) -> None:
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
        "cell_model_config": asdict(CELL_MODEL_CONFIG),
        "marker_model_config": asdict(MARKER_MODEL_CONFIG),
        "optional_region_model_config": asdict(OPTIONAL_REGION_MODEL_CONFIG),
        "colocalization_config": asdict(COLOCALIZATION_CONFIG),
        "runtime_config": asdict(RUNTIME_CONFIG),
    }
    (results_dir / f"{stem}_analysis_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

# %% MAIN FUNCTION
def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    ground_truth_dir = data_dir / "ground_truth"
    stack_paths = sorted(data_dir.glob("synthetic_stack_*.ome.tif"))
    if args.limit is not None:
        stack_paths = stack_paths[: args.limit]

    if not stack_paths:
        raise FileNotFoundError(
            f"No synthetic benchmark stacks found in {data_dir}. "
            "Run create_synthetic_benchmark_data.py first."
        )

    gt_summary = pd.read_csv(ground_truth_dir / "synthetic_benchmark_ground_truth_summary.csv")
    print(f"Loaded ground-truth summary for {len(gt_summary)} stacks.")

    for stack_path in stack_paths:
        # stack_path = stack_paths[0]  # for debugging
        overview_sidecar = stack_path.parent / "results" / f"{stack_path.stem}_roi_overview.csv"
        if overview_sidecar.exists() and not args.overwrite:
            print(f"Skipping existing analysis: {stack_path.name}")
            continue

        loaded_images = load_analysis_images(
            source_path=stack_path,
            channel_config=CHANNEL_CONFIG,
            voxel_scale_zyx=(1.0, 1.0),
            crop_for_testing=None,
            image_loading_mode=RUNTIME_CONFIG.image_loading_mode,
        )
        loaded_images = prepare_loaded_images_for_analysis(
            loaded_images,
            CELL_MODEL_CONFIG,
            MARKER_MODEL_CONFIG,
            OPTIONAL_REGION_MODEL_CONFIG,
        )

        roi_labels_2d = create_full_image_roi_labels(loaded_images.cell_image.shape[1:])
        run_result = run_roi_cellpose_colocalization(
            loaded_images=loaded_images,
            roi_labels_2d=roi_labels_2d,
            cell_model_config=CELL_MODEL_CONFIG,
            marker_model_config=MARKER_MODEL_CONFIG,
            colocalization_config=COLOCALIZATION_CONFIG,
            runtime_config=RUNTIME_CONFIG,
            optional_region_model_config=OPTIONAL_REGION_MODEL_CONFIG,
            optional_region_result=None,
        )
        export_analysis_outputs(
            run_result=run_result,
            paths=loaded_images.paths,
            optional_region_result=None,
        )
        write_table_sidecars(stack_path=stack_path, run_result=run_result)

        n_cells = int(run_result.tables.overview["n_cells"].sum())
        n_positive = int(run_result.tables.overview["n_marker_positive_cells"].sum())
        region_coverage = float(
            run_result.tables.overview["optional_region_occupancy_coverage_2d_percent"].mean()
        )
        print(
            f"Processed {stack_path.name}: "
            f"detected cells={n_cells}, "
            f"marker-positive={n_positive}, "
            f"optional-region coverage={region_coverage:.2f}%")
# %% MAIN ENTRY
if __name__ == "__main__":
    main()
# %% END
