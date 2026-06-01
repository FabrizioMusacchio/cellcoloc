"""napari visualization helpers for the interactive user workflow."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import DisplayNames
from .roi import get_roi_label_points
from .schemas import ColocalizationRunResult, LoadedImageChannels, OptionalRegionSegmentationResult


def _get_or_create_viewer(existing_viewer=None):
    """Return an existing napari viewer or create a new one."""

    import napari

    if _viewer_is_usable(existing_viewer):
        return existing_viewer

    current_viewer_getter = getattr(napari, "current_viewer", None)
    if callable(current_viewer_getter):
        current_viewer = current_viewer_getter()
        if _viewer_is_usable(current_viewer):
            return current_viewer

    return napari.Viewer()


def _viewer_is_usable(viewer) -> bool:
    """Return whether a napari viewer object is still safe to reuse.

    A closed viewer can remain bound to a Python variable even though its Qt
    window has already been deleted. Reusing such a stale viewer leads to
    inconsistent internal dims state and layer-addition crashes. napari removes
    closed viewers from ``Viewer._instances``, which gives us a reliable
    lightweight liveness check.
    """

    if viewer is None:
        return False

    viewer_instances = getattr(type(viewer), "_instances", None)
    if viewer_instances is None:
        return False

    if viewer not in viewer_instances:
        return False

    try:
        _ = viewer.layers
        _ = viewer.dims.ndim
    except Exception:
        return False

    return True


def _remove_layer_if_present(viewer, layer_name: str) -> None:
    """Remove a napari layer when it already exists in the viewer."""

    if layer_name in viewer.layers:
        viewer.layers.remove(viewer.layers[layer_name])


def _hide_layer_if_present(viewer, layer_name: str) -> None:
    """Hide a napari layer when it exists in the viewer."""

    if layer_name in viewer.layers:
        viewer.layers[layer_name].visible = False


def _normalize_layer_selection(
    layers_to_show: Sequence[str] | None,
) -> set[str] | None:
    """Normalize the optional set of layer keys that should be refreshed."""

    if layers_to_show is None:
        return None
    return {layer_name.strip() for layer_name in layers_to_show}


def _should_render_layer(
    selected_layers: set[str] | None,
    layer_key: str,
) -> bool:
    """Return whether a given logical layer should be rendered."""

    return selected_layers is None or layer_key in selected_layers


def _replace_or_add_image(
    viewer,
    *,
    replace_existing_layers: bool,
    name: str,
    data,
    **kwargs,
):
    """Update an existing image layer in place or add it when missing."""

    if name in viewer.layers:
        layer = viewer.layers[name]
        layer.data = data
        if "scale" in kwargs:
            layer.scale = kwargs["scale"]
        if "blending" in kwargs:
            layer.blending = kwargs["blending"]
        if "colormap" in kwargs:
            layer.colormap = kwargs["colormap"]
        layer.visible = True
        return

    viewer.add_image(data, name=name, **kwargs)


def _replace_or_add_labels(
    viewer,
    *,
    replace_existing_layers: bool,
    name: str,
    data,
    **kwargs,
):
    """Update an existing labels layer in place or add it when missing."""

    if name in viewer.layers:
        layer = viewer.layers[name]
        layer.data = data
        if "scale" in kwargs:
            layer.scale = kwargs["scale"]
        if "blending" in kwargs:
            layer.blending = kwargs["blending"]
        layer.visible = True
        return

    viewer.add_labels(data, name=name, **kwargs)


def _replace_or_add_points(
    viewer,
    *,
    replace_existing_layers: bool,
    name: str,
    data,
    **kwargs,
):
    """Update an existing points layer in place or add it when missing."""

    if name in viewer.layers:
        layer = viewer.layers[name]
        layer.data = data
        if "scale" in kwargs:
            layer.scale = kwargs["scale"]
        if "size" in kwargs:
            layer.size = kwargs["size"]
        if "face_color" in kwargs:
            layer.face_color = kwargs["face_color"]
        if "text" in kwargs:
            layer.text = kwargs["text"]
        layer.visible = True
        return

    viewer.add_points(data, name=name, **kwargs)


def extract_label_masks_from_viewer(
    viewer,
    cell_layer_name: str = "Cellpose cell masks",
    marker_layer_name: str = "Cellpose marker masks",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the current cell and marker label layers from a napari viewer."""

    if cell_layer_name not in viewer.layers:
        raise KeyError(f"Cell label layer not found in viewer: {cell_layer_name}")
    if marker_layer_name not in viewer.layers:
        raise KeyError(f"Marker label layer not found in viewer: {marker_layer_name}")

    cell_masks = np.asarray(viewer.layers[cell_layer_name].data, dtype=np.uint32)
    marker_masks = np.asarray(viewer.layers[marker_layer_name].data, dtype=np.uint32)
    return cell_masks, marker_masks


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
    layers_to_show: Sequence[str] | None = None,
    replace_existing_layers: bool = True,
    show_optional_region_image: bool = False,
):
    """Display or refresh analysis layers in napari.

    Parameters
    ----------
    layers_to_show:
        Optional list of logical layer keys to add or refresh. Supported keys
        are ``"cell_image"``, ``"marker_image"``, ``"optional_region_image"``,
        ``"optional_region_labels"``, ``"rois"``, ``"roi_numbers"``,
        ``"cell_masks"``, ``"marker_masks"``, and ``"positive_cells"``.
        When ``None``, the function renders all standard analysis layers.
    replace_existing_layers:
        If ``True``, existing layers with the same name are removed and added
        again. This keeps repeated refinement runs from piling up duplicate
        layers in the same viewer.
    show_optional_region_image:
        If ``True``, the optional third-channel image is shown even when no
        threshold result object is provided.
    """

    display_names = display_names or DisplayNames()
    viewer = _get_or_create_viewer(viewer)
    selected_layers = _normalize_layer_selection(layers_to_show)

    if _should_render_layer(selected_layers, "cell_image"):
        _replace_or_add_image(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=loaded_images.cell_image,
            name=display_names.cell,
            scale=loaded_images.voxel_scale_zyx,
            blending="additive",
            colormap="magenta",
            channel_axis=None,
        )
    if _should_render_layer(selected_layers, "marker_image"):
        _replace_or_add_image(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=loaded_images.marker_image,
            name=display_names.marker,
            scale=loaded_images.voxel_scale_zyx,
            blending="additive",
            colormap="cyan",
            channel_axis=None,
        )

    if loaded_images.optional_region_image is not None and (
        show_optional_region_image or optional_region_result is not None
    ):
        if _should_render_layer(selected_layers, "optional_region_image"):
            _replace_or_add_image(
                viewer,
                replace_existing_layers=replace_existing_layers,
                data=loaded_images.optional_region_image,
                name=display_names.optional_region,
                scale=loaded_images.voxel_scale_zyx,
                blending="additive",
                colormap="red",
                channel_axis=None,
            )
    if optional_region_result is not None and _should_render_layer(
        selected_layers,
        "optional_region_labels",
    ):
        _replace_or_add_labels(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=optional_region_result.labels,
            name=f"{display_names.optional_region} threshold labels",
            blending="additive",
            scale=loaded_images.voxel_scale_zyx,
        )

    roi_labels_3d = np.repeat(roi_labels_2d[np.newaxis, :, :], loaded_images.cell_image.shape[0], axis=0)
    if _should_render_layer(selected_layers, "rois"):
        _replace_or_add_labels(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=roi_labels_3d,
            name="ROIs",
            blending="additive",
            scale=loaded_images.voxel_scale_zyx,
        )

    roi_points_yx, roi_text_labels = get_roi_label_points(roi_labels_2d)
    if len(roi_points_yx) > 0 and _should_render_layer(selected_layers, "roi_numbers"):
        _replace_or_add_points(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=roi_points_yx,
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

    if _should_render_layer(selected_layers, "cell_masks"):
        _replace_or_add_labels(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=run_result.cell_masks,
            name="Cellpose cell masks",
            blending="additive",
            scale=loaded_images.voxel_scale_zyx,
        )
    if _should_render_layer(selected_layers, "marker_masks"):
        _replace_or_add_labels(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=run_result.marker_masks,
            name="Cellpose marker masks",
            blending="additive",
            scale=loaded_images.voxel_scale_zyx,
        )
    if _should_render_layer(selected_layers, "positive_cells"):
        _replace_or_add_labels(
            viewer,
            replace_existing_layers=replace_existing_layers,
            data=run_result.positive_cell_masks,
            name=display_names.positive_cells,
            blending="additive",
            scale=loaded_images.voxel_scale_zyx,
        )

    _hide_layer_if_present(viewer, f"{display_names.cell} max projection for ROI drawing")
    _hide_layer_if_present(viewer, f"{display_names.marker} max projection for ROI drawing")
    _hide_layer_if_present(viewer, f"{display_names.optional_region} max projection for ROI drawing")
    _hide_layer_if_present(viewer, "Draw ROIs here")
    return viewer
