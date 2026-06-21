"""
ROI helpers for napari-based interactive analysis workflows.

author: Fabrizio Musacchio
date: May/June 2026
"""
# %% IMPORTS
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from .config import DisplayNames
from .io import save_roi_labels
from .schemas import LoadedImageChannels

# %% ROI HELPERS
def rasterize_shapes_to_labelmask(
    shapes_layer,
    image_shape_yx: tuple[int, int],
    scale_yx: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Rasterize napari polygon-like shapes into a 2D label mask.

    Napari stores shape coordinates in world coordinates when image layers use a
    physical scale. The coordinates are therefore mapped back to pixel
    coordinates before rasterization.
    """

    roi_labels = np.zeros(image_shape_yx, dtype=np.uint16)
    yy_full, xx_full = np.mgrid[0:image_shape_yx[0], 0:image_shape_yx[1]]
    scale_y, scale_x = scale_yx

    for roi_id, shape in enumerate(shapes_layer.data, start=1):
        coords = np.asarray(shape)

        if coords.ndim != 2 or coords.shape[1] < 2:
            print(f"Skipping ROI {roi_id}: unexpected coordinate array {coords.shape}")
            continue

        y = coords[:, 0] / scale_y
        x = coords[:, 1] / scale_x

        y_min = max(int(np.floor(y.min())), 0)
        y_max = min(int(np.ceil(y.max())) + 1, image_shape_yx[0])
        x_min = max(int(np.floor(x.min())), 0)
        x_max = min(int(np.ceil(x.max())) + 1, image_shape_yx[1])

        if y_max <= y_min or x_max <= x_min:
            print(f"Skipping ROI {roi_id}: empty bounding box")
            continue

        yy = yy_full[y_min:y_max, x_min:x_max]
        xx = xx_full[y_min:y_max, x_min:x_max]
        points = np.column_stack([yy.ravel(), xx.ravel()])
        polygon = MplPath(np.column_stack([y, x]))
        mask = polygon.contains_points(points).reshape(yy.shape)
        roi_labels[y_min:y_max, x_min:x_max][mask] = roi_id

    return roi_labels


def create_roi_drawing_viewer(
    loaded_images: LoadedImageChannels,
    display_names: DisplayNames | None = None,
):
    """Open a napari viewer for drawing 2D ROIs on max projections."""

    import napari

    display_names = display_names or DisplayNames()
    projection_cell = loaded_images.cell_image.max(axis=0)
    projection_marker = loaded_images.marker_image.max(axis=0)

    viewer = napari.Viewer()
    viewer.add_image(
        projection_cell,
        name=f"{display_names.cell} max projection for ROI drawing",
        scale=loaded_images.voxel_scale_zyx[1:],
    )
    viewer.add_image(
        projection_marker,
        name=f"{display_names.marker} max projection for ROI drawing",
        scale=loaded_images.voxel_scale_zyx[1:],
        blending="additive",
    )

    if loaded_images.optional_region_image is not None:
        viewer.add_image(
            loaded_images.optional_region_image.max(axis=0),
            name=f"{display_names.optional_region} max projection for ROI drawing",
            scale=loaded_images.voxel_scale_zyx[1:],
            blending="additive",
        )

    shapes_layer = viewer.add_shapes(
        name="Draw ROIs here",
        ndim=2,
        shape_type="polygon",
        edge_width=2,
        face_color="transparent",
        blending="additive",
    )

    return viewer, shapes_layer


def save_roi_labels_from_shapes(
    shapes_layer,
    output_path: Path,
    image_shape_yx: tuple[int, int],
    scale_yx: tuple[float, float],
) -> np.ndarray:
    """Rasterize drawn ROIs and save them to the standard TIFF file."""

    roi_labels = rasterize_shapes_to_labelmask(
        shapes_layer=shapes_layer,
        image_shape_yx=image_shape_yx,
        scale_yx=scale_yx,
    )
    save_roi_labels(output_path, roi_labels)
    return roi_labels


def get_bbox_2d(mask_2d: np.ndarray) -> tuple[slice, slice] | None:
    """Return the bounding box slices of a 2D binary mask."""

    y, x = np.where(mask_2d)
    if len(y) == 0:
        return None

    return (slice(y.min(), y.max() + 1), slice(x.min(), x.max() + 1))


def get_roi_label_points(roi_labels_2d: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return ROI centroids and text labels for napari point annotations."""

    points: list[list[float]] = []
    labels: list[str] = []

    roi_ids = np.unique(roi_labels_2d)
    roi_ids = roi_ids[roi_ids != 0]

    for roi_id in roi_ids:
        yy, xx = np.where(roi_labels_2d == roi_id)
        if len(yy) == 0:
            continue

        points.append([float(np.mean(yy)), float(np.mean(xx))])
        labels.append(f"ROI {int(roi_id)}")

    if not points:
        return np.empty((0, 2), dtype=float), labels

    return np.asarray(points, dtype=float), labels


def create_full_image_roi_labels(image_shape_yx: tuple[int, int]) -> np.ndarray:
    """Create a single ROI label covering the complete field of view.

    This helper is useful for analyses that should be run on the entire image
    without manual ROI drawing. The returned label image contains the label
    value ``1`` everywhere.
    """

    return np.ones(image_shape_yx, dtype=np.uint16)
# %% END