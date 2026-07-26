"""Create synthetic three-channel benchmark datasets for the CellColoc preprint.

The generator supports two render profiles built from the same object supports:

- ``gaussian``: microscopy-like Gaussian intensity blobs
- ``filled``: sharp, fully filled object blobs

In both cases, the script also writes per-stack and per-object ground-truth
tables as well as labeled ground-truth masks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops
from skimage.morphology import closing, disk, opening, remove_small_holes, remove_small_objects


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "example_data" / "synthetic_benchmark_data"

IMAGE_SHAPE_YX = (256, 256)
N_STACKS = 20
RNG_SEED = 20260709

MIN_CHANNEL0_OBJECTS = 10
MAX_CHANNEL0_OBJECTS = 30
MIN_DIAMETER_PX = 15.0
MAX_DIAMETER_PX = 35.0

OVERLAP_FRACTION_TARGET = 0.70
CHANNEL2_TARGET_COVERAGE = 0.30

GT_MIN_OVERLAP_PIXELS = 10
GT_MIN_OVERLAP_FRACTION = 0.02


@dataclass(frozen=True)
class BlobSpec:
    center_y: float
    center_x: float
    sigma_y: float
    sigma_x: float
    angle_rad: float
    amplitude: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing TIFF stacks and ground-truth files.",
    )
    parser.add_argument(
        "--profile",
        choices=("gaussian", "filled"),
        default="gaussian",
        help="Intensity-rendering profile for channels 0 and 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Target dataset directory (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--n-stacks",
        type=int,
        default=N_STACKS,
        help=f"Number of synthetic stacks to generate (default: {N_STACKS}).",
    )
    return parser.parse_args()


def make_dirs(data_dir: Path, ground_truth_dir: Path, ground_truth_label_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_label_dir.mkdir(parents=True, exist_ok=True)


def rotated_gaussian_and_mask(
    spec: BlobSpec,
    image_shape: tuple[int, int],
    support_level: float = np.exp(-2.0),
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:image_shape[0], 0:image_shape[1]]
    dy = yy - spec.center_y
    dx = xx - spec.center_x
    cos_a = np.cos(spec.angle_rad)
    sin_a = np.sin(spec.angle_rad)
    yr = cos_a * dy + sin_a * dx
    xr = -sin_a * dy + cos_a * dx
    exponent = -0.5 * ((yr / spec.sigma_y) ** 2 + (xr / spec.sigma_x) ** 2)
    gaussian = spec.amplitude * np.exp(exponent)
    support_mask = gaussian >= float(spec.amplitude * support_level)
    return gaussian, support_mask


def masks_touch(mask_a: np.ndarray, mask_b: np.ndarray, margin: int = 3) -> bool:
    if not np.any(mask_a) or not np.any(mask_b):
        return False
    dilated = closing(mask_a, footprint=disk(max(1, margin)))
    return bool(np.any(dilated & mask_b))


def sample_channel0_specs(
    rng: np.random.Generator,
    n_objects: int,
    image_shape: tuple[int, int],
) -> list[BlobSpec]:
    specs: list[BlobSpec] = []
    masks: list[np.ndarray] = []
    for _ in range(n_objects):
        placed = False
        for _attempt in range(800):
            diameter = float(rng.uniform(MIN_DIAMETER_PX, MAX_DIAMETER_PX))
            sigma_major = diameter / 5.0
            sigma_minor = sigma_major * float(rng.uniform(0.65, 1.0))
            if rng.random() < 0.5:
                sigma_y, sigma_x = sigma_major, sigma_minor
            else:
                sigma_y, sigma_x = sigma_minor, sigma_major
            angle_rad = float(rng.uniform(0.0, np.pi))
            radius_y = int(np.ceil(3.0 * sigma_y))
            radius_x = int(np.ceil(3.0 * sigma_x))
            center_y = float(rng.uniform(radius_y + 4, image_shape[0] - radius_y - 4))
            center_x = float(rng.uniform(radius_x + 4, image_shape[1] - radius_x - 4))
            amplitude = float(rng.uniform(0.75, 1.00))
            candidate = BlobSpec(
                center_y=center_y,
                center_x=center_x,
                sigma_y=sigma_y,
                sigma_x=sigma_x,
                angle_rad=angle_rad,
                amplitude=amplitude,
            )
            _, candidate_mask = rotated_gaussian_and_mask(candidate, image_shape)
            if any(masks_touch(candidate_mask, existing_mask, margin=3) for existing_mask in masks):
                continue
            specs.append(candidate)
            masks.append(candidate_mask)
            placed = True
            break
        if not placed:
            raise RuntimeError("Could not place all synthetic channel-0 blobs without excessive overlap.")
    return specs


def find_non_overlapping_center(
    rng: np.random.Generator,
    image_shape: tuple[int, int],
    sigma_y: float,
    sigma_x: float,
    forbidden_masks: list[np.ndarray],
) -> tuple[float, float]:
    radius_y = int(np.ceil(3.0 * sigma_y))
    radius_x = int(np.ceil(3.0 * sigma_x))
    for _attempt in range(800):
        center_y = float(rng.uniform(radius_y + 4, image_shape[0] - radius_y - 4))
        center_x = float(rng.uniform(radius_x + 4, image_shape[1] - radius_x - 4))
        candidate = BlobSpec(
            center_y=center_y,
            center_x=center_x,
            sigma_y=sigma_y,
            sigma_x=sigma_x,
            angle_rad=0.0,
            amplitude=1.0,
        )
        _, candidate_mask = rotated_gaussian_and_mask(candidate, image_shape)
        if any(np.any(candidate_mask & forbidden) for forbidden in forbidden_masks):
            continue
        return center_y, center_x
    raise RuntimeError("Could not place a non-overlapping synthetic marker blob.")


def sample_channel1_specs(
    rng: np.random.Generator,
    channel0_specs: list[BlobSpec],
    image_shape: tuple[int, int],
) -> tuple[list[BlobSpec], list[int]]:
    n_objects = len(channel0_specs)
    n_overlapping = int(round(OVERLAP_FRACTION_TARGET * n_objects))
    overlapping_indices = sorted(rng.choice(n_objects, size=n_overlapping, replace=False).tolist())

    channel0_masks = [
        rotated_gaussian_and_mask(spec, image_shape)[1]
        for spec in channel0_specs
    ]

    marker_specs: list[BlobSpec] = []
    marker_masks: list[np.ndarray] = []
    for object_index, base_spec in enumerate(channel0_specs):
        base_diameter = 5.0 * max(base_spec.sigma_y, base_spec.sigma_x)
        sigma_major = float(rng.uniform(0.85, 1.10) * (base_diameter / 5.0))
        sigma_minor = sigma_major * float(rng.uniform(0.65, 1.00))
        if rng.random() < 0.5:
            sigma_y, sigma_x = sigma_major, sigma_minor
        else:
            sigma_y, sigma_x = sigma_minor, sigma_major
        angle_rad = float(rng.uniform(0.0, np.pi))
        amplitude = float(rng.uniform(0.72, 0.98))

        if object_index in overlapping_indices:
            placed = False
            for _attempt in range(300):
                shift_scale = float(rng.uniform(0.0, 0.35 * base_diameter))
                shift_angle = float(rng.uniform(0.0, 2.0 * np.pi))
                center_y = float(base_spec.center_y + np.sin(shift_angle) * shift_scale)
                center_x = float(base_spec.center_x + np.cos(shift_angle) * shift_scale)
                radius_y = int(np.ceil(3.0 * sigma_y))
                radius_x = int(np.ceil(3.0 * sigma_x))
                if not (radius_y + 4 <= center_y <= image_shape[0] - radius_y - 4):
                    continue
                if not (radius_x + 4 <= center_x <= image_shape[1] - radius_x - 4):
                    continue
                candidate = BlobSpec(
                    center_y=center_y,
                    center_x=center_x,
                    sigma_y=sigma_y,
                    sigma_x=sigma_x,
                    angle_rad=angle_rad,
                    amplitude=amplitude,
                )
                _, candidate_mask = rotated_gaussian_and_mask(candidate, image_shape)
                if not np.any(candidate_mask & channel0_masks[object_index]):
                    continue
                marker_specs.append(candidate)
                marker_masks.append(candidate_mask)
                placed = True
                break
            if not placed:
                raise RuntimeError("Could not place an overlapping synthetic marker blob.")
        else:
            center_y, center_x = find_non_overlapping_center(
                rng=rng,
                image_shape=image_shape,
                sigma_y=sigma_y,
                sigma_x=sigma_x,
                forbidden_masks=channel0_masks + marker_masks,
            )
            candidate = BlobSpec(
                center_y=center_y,
                center_x=center_x,
                sigma_y=sigma_y,
                sigma_x=sigma_x,
                angle_rad=angle_rad,
                amplitude=amplitude,
            )
            _, candidate_mask = rotated_gaussian_and_mask(candidate, image_shape)
            marker_specs.append(candidate)
            marker_masks.append(candidate_mask)

    return marker_specs, overlapping_indices


def build_label_image_from_specs(
    specs: list[BlobSpec],
    image_shape: tuple[int, int],
    profile: str,
) -> tuple[np.ndarray, np.ndarray]:
    intensity = np.zeros(image_shape, dtype=np.float32)
    labels = np.zeros(image_shape, dtype=np.uint16)
    for label_id, spec in enumerate(specs, start=1):
        gaussian, support_mask = rotated_gaussian_and_mask(spec, image_shape)
        if profile == "gaussian":
            rendered_object = gaussian
        elif profile == "filled":
            rendered_object = support_mask.astype(np.float32) * float(spec.amplitude)
        else:
            raise ValueError(f"Unsupported render profile: {profile!r}")
        intensity = np.maximum(intensity, rendered_object)
        labels[support_mask] = label_id
    return intensity, labels


def build_channel2_mask(rng: np.random.Generator, image_shape: tuple[int, int]) -> np.ndarray:
    coarse = gaussian_filter(rng.normal(size=image_shape), sigma=18.0)
    threshold = float(np.quantile(coarse, 1.0 - CHANNEL2_TARGET_COVERAGE))
    mask = coarse >= threshold
    mask = opening(mask, footprint=disk(3))
    mask = closing(mask, footprint=disk(6))
    mask = remove_small_objects(mask, 350)
    mask = remove_small_holes(mask, 300)
    coverage = float(mask.mean())
    if coverage <= 0.0:
        raise RuntimeError("Synthetic channel-2 coverage mask became empty.")
    if abs(coverage - CHANNEL2_TARGET_COVERAGE) > 0.06:
        corrected_threshold = float(np.quantile(coarse, 1.0 - CHANNEL2_TARGET_COVERAGE))
        mask = coarse >= corrected_threshold
        mask = closing(mask, footprint=disk(4))
        mask = remove_small_holes(mask, 200)
    return mask.astype(bool)


def add_noise_and_scale(
    rng: np.random.Generator,
    image: np.ndarray,
    mask: np.ndarray | None = None,
    noise_sigma: float = 0.035,
) -> np.ndarray:
    noisy = np.asarray(image, dtype=np.float32)
    if mask is not None:
        noisy = noisy + mask.astype(np.float32) * float(rng.uniform(0.55, 0.75))
        noisy = gaussian_filter(noisy, sigma=1.0)
    noisy = noisy + rng.normal(loc=0.05, scale=noise_sigma, size=noisy.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, None)
    if float(noisy.max()) > 0:
        noisy = noisy / float(noisy.max())
    return (noisy * 65535).astype(np.uint16)


def compute_roundness(area_px: float, perimeter_px: float) -> float:
    if not np.isfinite(perimeter_px) or perimeter_px <= 0:
        return float("nan")
    return float((4.0 * np.pi * area_px) / (perimeter_px ** 2))


def build_object_table(
    stack_id: str,
    channel_name: str,
    labels: np.ndarray,
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    for region in regionprops(labels):
        area_px = float(region.area)
        perimeter_px = float(getattr(region, "perimeter", np.nan))
        rows.append(
            {
                "stack_id": stack_id,
                "channel": channel_name,
                "label": int(region.label),
                "centroid_y": float(region.centroid[0]),
                "centroid_x": float(region.centroid[1]),
                "area_px_2d": area_px,
                "perimeter_px_2d": perimeter_px,
                "roundness_2d": compute_roundness(area_px, perimeter_px),
                "eccentricity_2d": float(getattr(region, "eccentricity", np.nan)),
            }
        )
    return rows


def summarize_overlap_gt(
    channel0_labels: np.ndarray,
    channel1_labels: np.ndarray,
) -> tuple[list[dict[str, float | int | str | bool]], int]:
    rows: list[dict[str, float | int | str | bool]] = []
    positive_count = 0
    for region in regionprops(channel0_labels):
        cell_mask = channel0_labels == region.label
        overlapping_labels = channel1_labels[cell_mask]
        overlapping_labels = overlapping_labels[overlapping_labels != 0]
        if overlapping_labels.size == 0:
            best_marker_label = np.nan
            best_overlap_pixels = 0
            best_overlap_fraction = 0.0
            marker_positive = False
        else:
            unique_labels, counts = np.unique(overlapping_labels, return_counts=True)
            best_idx = int(np.argmax(counts))
            best_marker_label = int(unique_labels[best_idx])
            best_overlap_pixels = int(counts[best_idx])
            best_overlap_fraction = float(best_overlap_pixels / float(region.area))
            marker_positive = (
                best_overlap_pixels >= GT_MIN_OVERLAP_PIXELS
                and best_overlap_fraction >= GT_MIN_OVERLAP_FRACTION
            )
        positive_count += int(marker_positive)
        rows.append(
            {
                "channel0_label": int(region.label),
                "best_channel1_label": best_marker_label,
                "best_overlap_pixels": best_overlap_pixels,
                "best_overlap_fraction": best_overlap_fraction,
                "marker_positive_gt": bool(marker_positive),
            }
        )
    return rows, positive_count


def save_ome_tiff(path: Path, stack_cyx: np.ndarray) -> None:
    data_tzcyx = stack_cyx[np.newaxis, np.newaxis, :, :, :]
    tifffile.imwrite(
        path,
        data_tzcyx,
        ome=True,
        metadata={
            "axes": "TZCYX",
            "PhysicalSizeX": 1.0,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 1.0,
            "PhysicalSizeYUnit": "µm",
        },
    )


def main() -> None:
    args = parse_args()
    data_dir = args.output_dir.resolve()
    ground_truth_dir = data_dir / "ground_truth"
    ground_truth_label_dir = ground_truth_dir / "labels"
    make_dirs(data_dir, ground_truth_dir, ground_truth_label_dir)
    rng = np.random.default_rng(RNG_SEED)

    object_rows: list[dict[str, float | int | str | bool]] = []
    overlap_rows: list[dict[str, float | int | str | bool]] = []
    summary_rows: list[dict[str, float | int | str | bool]] = []

    for stack_index in range(args.n_stacks):
        stack_id = f"synthetic_stack_{stack_index:02d}"
        stack_path = data_dir / f"{stack_id}.ome.tif"
        channel0_gt_path = ground_truth_label_dir / f"{stack_id}_channel0_labels.tif"
        channel1_gt_path = ground_truth_label_dir / f"{stack_id}_channel1_labels.tif"
        channel2_gt_path = ground_truth_label_dir / f"{stack_id}_channel2_mask.tif"

        if not args.overwrite and stack_path.exists():
            print(f"Skipping existing stack: {stack_path.name}")
            continue

        n_channel0 = int(rng.integers(MIN_CHANNEL0_OBJECTS, MAX_CHANNEL0_OBJECTS + 1))
        channel0_specs = sample_channel0_specs(rng, n_channel0, IMAGE_SHAPE_YX)
        channel1_specs, overlapping_indices = sample_channel1_specs(rng, channel0_specs, IMAGE_SHAPE_YX)

        channel0_signal, channel0_labels = build_label_image_from_specs(
            channel0_specs,
            IMAGE_SHAPE_YX,
            profile=args.profile,
        )
        channel1_signal, channel1_labels = build_label_image_from_specs(
            channel1_specs,
            IMAGE_SHAPE_YX,
            profile=args.profile,
        )

        channel2_mask = build_channel2_mask(rng, IMAGE_SHAPE_YX)
        channel2_signal = add_noise_and_scale(rng, np.zeros(IMAGE_SHAPE_YX, dtype=np.float32), mask=channel2_mask)

        channel0_image = add_noise_and_scale(rng, channel0_signal)
        channel1_image = add_noise_and_scale(rng, channel1_signal)

        stack_cyx = np.stack([channel0_image, channel1_image, channel2_signal], axis=0)
        save_ome_tiff(stack_path, stack_cyx)

        tifffile.imwrite(channel0_gt_path, channel0_labels.astype(np.uint16))
        tifffile.imwrite(channel1_gt_path, channel1_labels.astype(np.uint16))
        tifffile.imwrite(channel2_gt_path, channel2_mask.astype(np.uint8))

        channel0_rows = build_object_table(stack_id, "channel0", channel0_labels)
        channel1_rows = build_object_table(stack_id, "channel1", channel1_labels)
        for row in channel0_rows + channel1_rows:
            object_rows.append(row)

        stack_overlap_rows, n_channel0_marker_positive = summarize_overlap_gt(
            channel0_labels=channel0_labels,
            channel1_labels=channel1_labels,
        )
        for row in stack_overlap_rows:
            overlap_rows.append({"stack_id": stack_id, **row})

        channel2_coverage_percent = float(100.0 * channel2_mask.mean())
        summary_rows.append(
            {
                "stack_id": stack_id,
                "image_height_px": IMAGE_SHAPE_YX[0],
                "image_width_px": IMAGE_SHAPE_YX[1],
                "image_area_px": int(IMAGE_SHAPE_YX[0] * IMAGE_SHAPE_YX[1]),
                "render_profile": args.profile,
                "n_channel0_cells_gt": int(channel0_labels.max()),
                "n_channel1_cells_gt": int(channel1_labels.max()),
                "n_channel0_marker_positive_gt": int(n_channel0_marker_positive),
                "n_channel0_marker_negative_gt": int(channel0_labels.max() - n_channel0_marker_positive),
                "n_channel1_seeded_overlapping_objects": int(len(overlapping_indices)),
                "n_channel1_seeded_non_overlapping_objects": int(len(channel1_specs) - len(overlapping_indices)),
                "channel2_coverage_gt_percent": channel2_coverage_percent,
            }
        )

        print(
            f"Generated {stack_path.name}: "
            f"channel0={int(channel0_labels.max())} cells, "
            f"channel1={int(channel1_labels.max())} cells, "
            f"channel0 positives={n_channel0_marker_positive}, "
            f"channel2 coverage={channel2_coverage_percent:.1f}%"
        )

    pd.DataFrame(summary_rows).sort_values("stack_id").to_csv(
        ground_truth_dir / "synthetic_benchmark_ground_truth_summary.csv",
        index=False,
    )
    pd.DataFrame(object_rows).sort_values(["stack_id", "channel", "label"]).to_csv(
        ground_truth_dir / "synthetic_benchmark_ground_truth_objects.csv",
        index=False,
    )
    pd.DataFrame(overlap_rows).sort_values(["stack_id", "channel0_label"]).to_csv(
        ground_truth_dir / "synthetic_benchmark_ground_truth_colocalization.csv",
        index=False,
    )
    print(f"Saved synthetic benchmark data ({args.profile}) to:\n{data_dir}")


if __name__ == "__main__":
    main()
