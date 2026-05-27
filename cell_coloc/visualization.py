"""napari visualization helpers for the interactive user workflow."""

from __future__ import annotations

import numpy as np

from .config import DisplayNames
from .roi import get_roi_label_points
from .schemas import ColocalizationRunResult, LoadedImageChannels, OptionalRegionSegmentationResult


def _get_or_create_viewer(existing_viewer=None):
    """Return an existing napari viewer or create a new one."""

    import napari

    return existing_viewer if existing_viewer is not None else napari.Viewer()


def _hide_layer_if_present(viewer, layer_name: str) -> None:
    """Hide a napari layer when it exists in the viewer."""

    if layer_name in viewer.layers:
        viewer.layers[layer_name].visible = False


def show_optional_region_segmentation(
    loaded_images: LoadedImageChannels,
    region_result: OptionalRegionSegmentationResult,
    roi_labels_2d: np.ndarray | None = None,
    display_names: DisplayNames | None = None,
    viewer=None,
):
    """Display the optional third-channel segmentation in napari."""

    display_names = display_names or DisplayNames()
    viewer = _get_or_create_viewer(viewer)

    viewer.add_image(
        loaded_images.optional_region_image,
        name=display_names.optional_region,
        scale=loaded_images.voxel_scale_zyx,
        blending="additive",
        colormap="red",
        channel_axis=None,
    )
    viewer.add_image(
        region_result.corrected_image,
        name=f"{display_names.optional_region} background corrected",
        scale=loaded_images.voxel_scale_zyx,
        blending="additive",
        colormap="yellow",
        channel_axis=None,
    )
    viewer.add_labels(
        region_result.labels,
        name=f"{display_names.optional_region} threshold labels",
        blending="additive",
        scale=loaded_images.voxel_scale_zyx,
    )

    if roi_labels_2d is not None:
        roi_labels_3d = np.repeat(roi_labels_2d[np.newaxis, :, :], loaded_images.cell_image.shape[0], axis=0)
        viewer.add_labels(
            roi_labels_3d,
            name="ROIs",
            blending="additive",
            scale=loaded_images.voxel_scale_zyx,
        )

    _hide_layer_if_present(viewer, f"{display_names.cell} max projection for ROI drawing")
    _hide_layer_if_present(viewer, f"{display_names.marker} max projection for ROI drawing")
    _hide_layer_if_present(viewer, f"{display_names.optional_region} max projection for ROI drawing")
    _hide_layer_if_present(viewer, "Draw ROIs here")
    return viewer


def show_analysis_results(
    loaded_images: LoadedImageChannels,
    roi_labels_2d: np.ndarray,
    run_result: ColocalizationRunResult,
    display_names: DisplayNames | None = None,
    optional_region_result: OptionalRegionSegmentationResult | None = None,
    viewer=None,
):
    """Display the final analysis layers in napari."""

    display_names = display_names or DisplayNames()
    viewer = _get_or_create_viewer(viewer)

    viewer.add_image(
        loaded_images.cell_image,
        name=display_names.cell,
        scale=loaded_images.voxel_scale_zyx,
        blending="additive",
        colormap="magenta",
        channel_axis=None,
    )
    viewer.add_image(
        loaded_images.marker_image,
        name=display_names.marker,
        scale=loaded_images.voxel_scale_zyx,
        blending="additive",
        colormap="cyan",
        channel_axis=None,
    )

    if optional_region_result is not None and loaded_images.optional_region_image is not None:
        viewer.add_image(
            loaded_images.optional_region_image,
            name=display_names.optional_region,
            scale=loaded_images.voxel_scale_zyx,
            blending="additive",
            colormap="red",
            channel_axis=None,
        )
        viewer.add_labels(
            optional_region_result.labels,
            name=f"{display_names.optional_region} threshold labels",
            blending="additive",
            scale=loaded_images.voxel_scale_zyx,
        )

    roi_labels_3d = np.repeat(roi_labels_2d[np.newaxis, :, :], loaded_images.cell_image.shape[0], axis=0)
    viewer.add_labels(
        roi_labels_3d,
        name="ROIs",
        blending="additive",
        scale=loaded_images.voxel_scale_zyx,
    )

    roi_points_yx, roi_text_labels = get_roi_label_points(roi_labels_2d)
    if len(roi_points_yx) > 0:
        viewer.add_points(
            roi_points_yx,
            name="ROI numbers",
            scale=loaded_images.voxel_scale_zyx[1:],
            size=10,
            face_color="transparent",
            text={
                "string": roi_text_labels,
                "size": 14,
                "color": "white",
                "anchor": "center",
            },
        )

    viewer.add_labels(
        run_result.cell_masks,
        name="Cellpose cell masks",
        blending="additive",
        scale=loaded_images.voxel_scale_zyx,
    )
    viewer.add_labels(
        run_result.marker_masks,
        name="Cellpose marker masks",
        blending="additive",
        scale=loaded_images.voxel_scale_zyx,
    )
    viewer.add_labels(
        run_result.positive_cell_masks,
        name=display_names.positive_cells,
        blending="additive",
        scale=loaded_images.voxel_scale_zyx,
    )

    _hide_layer_if_present(viewer, f"{display_names.cell} max projection for ROI drawing")
    _hide_layer_if_present(viewer, f"{display_names.marker} max projection for ROI drawing")
    _hide_layer_if_present(viewer, f"{display_names.optional_region} max projection for ROI drawing")
    _hide_layer_if_present(viewer, "Draw ROIs here")
    return viewer
