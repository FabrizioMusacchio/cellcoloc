"""
This script is a prototype for analyzing colocalization of neurons and nuclei in 3D microscopy 
images stored in CZI format, using Cellpose for segmentation and napari for ROI drawing and 
visualization.

Installation:
conda create -n cellpose_coloc python=3.12 -y
conda activate cellpose_coloc
conda install -y ipykernel
pip install omio-microscopy cellpose matplotlib


On windows, you may first need to install PyTorch with CUDA support 
(if you have a compatible NVIDIA GPU) to use Cellpose with GPU acceleration:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
Then, follow same steps as above. 

author: Fabrizio Musacchio
date: April/May 2026
"""
# %% IMPORTS
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import napari
from matplotlib.path import Path as MplPath
from cellpose import models

from skimage.measure import regionprops_table

import czifile

from skimage.filters import gaussian, threshold_otsu, threshold_li
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, ball
from skimage.measure import label
# %% SETTINGS
# ADJUST HERE: path to your CZI file
czi_path = Path(r"k:\Tmp\ID24137_3rdsection_contralateralCtx_DAPI-GB-NeuN_20x.czi")

# ADJUST HERE: zero-based channel numbers for neurons, DAPI, and cancer channels in the CZI file
neuron_channel_number = 2
dapi_channel_number   = 1
cancer_channel_number = 0

# ADJUST HERE: physical pixel sizes in microns for z, y, x dimensions:
VOXEL_SCALE_ZYX = (3.0, 0.3899, 0.3899) # pixel sizes in z, y, x in microns

# Cellpose parameters:
NEURON_DIAMETER = 60 # Cellpose diameter parameter for neuron segmentation
NUCLEI_DIAMETER = 20 # Cellpose diameter parameter for nuclei segmentation

# Colocalization parameters:
MIN_NEURON_VOXELS = 200 # define minimum neuron size in voxels to exclude small objects (e.g. debris)

COLOC_THRESHOLD = 0.02 # minimum fraction of neuron voxels that must overlap with a nucleus to be considered "DAPI-positive"
MIN_OVERLAP_VOXELS = 20
# A neuron is considered DAPI-positive if:
# 1. it overlaps with at least one DAPI/nucleus label,
# 2. the best overlap has at least MIN_OVERLAP_VOXELS voxels,
# 3. the best overlap fraction is at least COLOC_THRESHOLD.
# With the current logic, COLOC_THRESHOLD may be 0.0, because zero-overlap
# neurons are excluded by n_overlapping_nuclei > 0 and MIN_OVERLAP_VOXELS.

# Cancer segmentation parameters:
CANCER_SEGMENTATION_METHOD = "li"  # "otsu", "li", "percentile"
CANCER_THRESHOLD_PERCENTILE = 98.0 # only used if method=="percentile"
CANCER_GAUSSIAN_SIGMA = 1.0         # if >0, apply Gaussian smoothing with this sigma before thresholding
CANCER_BACKGROUND_SIGMA = None      # if >0, apply background subtraction using a Gaussian blur with this sigma
CANCER_MIN_OBJECT_VOXELS = 10       # remove small objects smaller than this size in voxels after thresholding
CANCER_MIN_HOLE_VOXELS = 10         # remove small holes smaller than this size in voxels after thresholding
CANCER_APPLY_CLOSING = True         # if True, apply binary closing to the cancer mask after thresholding to close small gaps


# further Cellpose and analysis parameters:
DRAW_ROIS = True          # True: draw ROIs in napari and save ROI label TIFF
PROCESS_ROIS = True       # True: run Cellpose inside all saved ROIs
OPEN_RESULTS = True       # True: open images, masks, ROI labels in napari at end
GPU = True                # True to use GPU for Cellpose (if available), False to use CPU
CROP_FOR_TESTING = None   # e.g. (slice(None), slice(0, 200), slice(0, 200)) or None to disable cropping

# DON'T ADJUST BELOW UNLESS YOU KNOW WHAT YOU ARE DOING:
roi_tif_path = czi_path.with_name(czi_path.stem + "_ROIs_labelmask.tif")
csv_out_path = czi_path.with_name(czi_path.stem + "_cellpose_colocalization.csv")

neuron_mask_out_path = czi_path.with_name(czi_path.stem + "_cellpose_neuron_masks.tif")
nuclei_mask_out_path = czi_path.with_name(czi_path.stem + "_cellpose_nuclei_masks.tif")
cancer_mask_out_path = czi_path.with_name(czi_path.stem + "_otsu_cancer_mask.tif")
# %% HELPERS

def load_czi_channel_as_zyx(czi_path: Path, channel_number: int) -> np.ndarray:
    """
    Load one zero-based CZI channel and return it as ZYX.
    """

    with czifile.CziFile(czi_path) as czi:
        axes = czi.axes
        arr = czi.asarray()

    print(f"\nLoaded CZI: {czi_path}")
    print(f"CZI axes:  {axes}")
    print(f"CZI shape: {arr.shape}")
    print(f"CZI dtype: {arr.dtype}")

    if "C" not in axes:
        raise ValueError(f"No channel axis 'C' found in CZI axes: {axes}")

    c_axis = axes.index("C")

    if channel_number < 0 or channel_number >= arr.shape[c_axis]:
        raise ValueError(f"Requested channel {channel_number}, but C axis has size {arr.shape[c_axis]}.")

    arr_ch = np.take(arr, indices=channel_number, axis=c_axis)
    axes_ch = axes[:c_axis] + axes[c_axis + 1:]

    kept_slices = []
    kept_axes = []

    for axis, size in zip(axes_ch, arr_ch.shape):
        if size == 1 and axis not in "ZYX":
            kept_slices.append(0)
        else:
            kept_slices.append(slice(None))
            kept_axes.append(axis)

    arr_ch = arr_ch[tuple(kept_slices)]
    kept_axes = "".join(kept_axes)

    print(f"Channel {channel_number} axes after singleton removal:  {kept_axes}")
    print(f"Channel {channel_number} shape after singleton removal: {arr_ch.shape}")

    if not all(axis in kept_axes for axis in "ZYX"):
        raise ValueError(f"Cannot construct ZYX. Available axes: {kept_axes}, shape: {arr_ch.shape}")

    arr_zyx = np.moveaxis(arr_ch, (kept_axes.index("Z"), kept_axes.index("Y"), kept_axes.index("X")), (0, 1, 2))

    print(f"Final channel {channel_number} ZYX shape: {arr_zyx.shape}")
    print(f"Final channel {channel_number} dtype:      {arr_zyx.dtype}")

    return np.asarray(arr_zyx)

def to_zyx(img: np.ndarray) -> np.ndarray:
    print(f"Original shape: {img.shape}")
    print(f"Original dtype: {img.dtype}")
    print(f"Original ndim:  {img.ndim}")

    img = np.asarray(img)
    squeezed = np.squeeze(img)

    print(f"Squeezed shape: {squeezed.shape}")

    if squeezed.ndim == 3:
        zyx = squeezed
    elif squeezed.ndim == 2:
        zyx = squeezed[np.newaxis, :, :]
    else:
        raise ValueError(
            f"Cannot safely convert image with shape {img.shape} "
            f"to ZYX after squeezing to {squeezed.shape}."
        )

    print(f"Final ZYX shape: {zyx.shape}")
    print(f"Final dtype:     {zyx.dtype}")

    return zyx

def rasterize_shapes_to_labelmask(shapes_layer, image_shape_yx, scale_yx=(1.0, 1.0)):
    """
    Rasterize napari polygon/rectangle/ellipse-like shapes into a 2D label image.

    Napari shape coordinates may be stored in world coordinates when the image
    layer has a physical scale. Therefore, coordinates are converted back to
    pixel coordinates by dividing by scale_yx.

    Assumption:
        ROIs are drawn on a 2D projection image with axes YX.
    """

    roi_labels = np.zeros(image_shape_yx, dtype=np.uint16)

    yy_full, xx_full = np.mgrid[0:image_shape_yx[0], 0:image_shape_yx[1]]

    scale_y, scale_x = scale_yx

    for roi_id, shape in enumerate(shapes_layer.data, start=1):
        coords = np.asarray(shape)

        if coords.ndim != 2 or coords.shape[1] < 2:
            print(f"Skipping ROI {roi_id}: unexpected shape coordinates {coords.shape}")
            continue

        # Convert world coordinates back to pixel coordinates.
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

def get_bbox_2d(mask_2d):
    y, x = np.where(mask_2d)

    if len(y) == 0:
        return None

    return (
        slice(y.min(), y.max() + 1),
        slice(x.min(), x.max() + 1),
    )

def relabel_with_offset(mask, offset):
    out = mask.copy()
    valid = out > 0
    out[valid] += offset
    return out

def filter_labels_by_size(label_img, min_size):
    labels, counts = np.unique(label_img, return_counts=True)
    keep_labels = labels[(labels != 0) & (counts >= min_size)]

    lut = np.zeros(int(label_img.max()) + 1, dtype=label_img.dtype)
    lut[keep_labels] = keep_labels

    return lut[label_img]

def analyze_colocalization(masks_neurons, masks_nuclei, roi_id):
    rows = []

    neuron_labels = np.unique(masks_neurons)
    neuron_labels = neuron_labels[neuron_labels != 0]

    for neuron_label in neuron_labels:
        neuron_mask = masks_neurons == neuron_label
        neuron_voxels = int(neuron_mask.sum())

        overlapping_nuclei = masks_nuclei[neuron_mask]
        overlapping_nuclei = overlapping_nuclei[overlapping_nuclei != 0]

        unique_nuclei, counts = np.unique(overlapping_nuclei, return_counts=True)

        if len(unique_nuclei) == 0:
            rows.append({
                "roi_id": roi_id,
                "neuron_label": int(neuron_label),
                "neuron_voxels": neuron_voxels,
                "n_overlapping_nuclei": 0,
                "nucleus_label": np.nan,
                "overlap_voxels": 0,
                "overlap_fraction_of_neuron": 0.0,
            })
        else:
            for nucleus_label, overlap_voxels in zip(unique_nuclei, counts):
                rows.append({
                    "roi_id": roi_id,
                    "neuron_label": int(neuron_label),
                    "neuron_voxels": neuron_voxels,
                    "n_overlapping_nuclei": int(len(unique_nuclei)),
                    "nucleus_label": int(nucleus_label),
                    "overlap_voxels": int(overlap_voxels),
                    "overlap_fraction_of_neuron": float(overlap_voxels / neuron_voxels),
                })

    return rows

def segment_cancer_channel(img_cancer, roi_labels_2d=None, method="percentile", percentile=99.0, gaussian_sigma=1.0, background_sigma=12.0, min_object_voxels=500, min_hole_voxels=500, apply_closing=True):
    """ 
    gaussian_sigma: if >0, apply Gaussian smoothing with this sigma before thresholding
    background_sigma: if >0, apply background subtraction using a Gaussian blur with this sigma
    
    DEBUG
    method          ="li"
    percentile      = 98.0
    gaussian_sigma  = 1
    background_sigma=None
    min_object_voxels=10
    min_hole_voxels =10
    apply_closing   =True
    """
    print("\nSegmenting cancer channel...")
    img = img_cancer.astype(np.float32)

    # prepare ROI mask:
    roi_mask_3d = None
    if roi_labels_2d is not None:
        roi_mask_3d = np.repeat((roi_labels_2d > 0)[np.newaxis, :, :], img.shape[0], axis=0)

    img_work = img.copy()
    if roi_mask_3d is not None:
        img_work[~roi_mask_3d] = 0

    # image pre-processing: gaussian smoothing (optional):
    if gaussian_sigma is not None and gaussian_sigma > 0:
        print(f"    Cancer Segmentation: Applying Gaussian smoothing with sigma={gaussian_sigma}...")
        img_smooth = gaussian(img_work, sigma=gaussian_sigma, preserve_range=True)
        if roi_mask_3d is not None:
            img_smooth[~roi_mask_3d] = 0
    else:
        img_smooth = img_work

    # image pre-processing: background subtraction (optional):
    if background_sigma is not None and background_sigma > 0:
        print(f"    Cancer Segmentation: Applying background subtraction with sigma={background_sigma}...")
        background = gaussian(img_smooth, sigma=background_sigma, preserve_range=True)
        img_corr = img_smooth - background
        img_corr[img_corr < 0] = 0
        if roi_mask_3d is not None:
            img_corr[~roi_mask_3d] = 0
    else:
        img_corr = img_smooth

    """ # TMP DEBUG:
    viewer.add_image(img_corr, 
                     name="Cancer channel background corrected", 
                     scale=VOXEL_SCALE_ZYX, 
                     blending="additive",
                     colormap="cyan",
                     channel_axis=None) """

    # cancer segmentation:
    print(f"    Cancer Segmentation: Computing threshold with method='{method}'...")
    if roi_mask_3d is not None:
        values = img_corr[roi_mask_3d]
    else:
        values = img_corr.ravel()

    values = values[np.isfinite(values)]
    values = values[values > 0]

    if len(values) == 0:
        raise ValueError("Cannot compute cancer threshold: no positive values found.")

    if method == "otsu":
        threshold = threshold_otsu(values)
    elif method == "li":
        threshold = threshold_li(values)
    elif method == "percentile":
        threshold = np.percentile(values, percentile)
    else:
        raise ValueError(f"Unknown cancer segmentation method: {method}")

    cancer_mask = np.zeros_like(img_corr, dtype=bool)

    if roi_mask_3d is not None:
        cancer_mask[roi_mask_3d] = img_corr[roi_mask_3d] > threshold
    else:
        cancer_mask = img_corr > threshold

    if apply_closing:
        print("    Cancer Segmentation: Applying binary closing to cancer mask...")
        cancer_mask = binary_closing(cancer_mask, footprint=ball(1))
        if roi_mask_3d is not None:
            cancer_mask[~roi_mask_3d] = False

    print(f"    Cancer Segmentation: Removing small objects smaller than {min_object_voxels} voxels...")
    cancer_mask = remove_small_objects(cancer_mask, min_size=min_object_voxels)

    print(f"    Cancer Segmentation: Removing small holes smaller than {min_hole_voxels} voxels...")
    cancer_mask = remove_small_holes(cancer_mask, area_threshold=min_hole_voxels)

    if roi_mask_3d is not None:
        cancer_mask[~roi_mask_3d] = False

    cancer_labels = label(cancer_mask).astype(np.uint32)

    """ # TMP DEBUG:
    viewer.add_labels(cancer_labels, 
                      name="Cancer Otsu labels", 
                      blending="additive", 
                      scale=VOXEL_SCALE_ZYX) """

    print(f"Method: {method}")
    print(f"Threshold: {threshold}")
    print(f"N cancer objects: {int(cancer_labels.max())}")
    print("...done.")

    return cancer_mask, cancer_labels, threshold, img_corr

def get_roi_label_points(roi_labels_2d):
    points = []
    labels = []

    roi_ids = np.unique(roi_labels_2d)
    roi_ids = roi_ids[roi_ids != 0]

    for roi_id in roi_ids:
        yy, xx = np.where(roi_labels_2d == roi_id)

        if len(yy) == 0:
            continue

        y = float(np.mean(yy))
        x = float(np.mean(xx))

        points.append([y, x])
        labels.append(f"ROI {roi_id}")

    return np.asarray(points), labels
# %% LOAD IMAGES FROM CZI
img_neurons = load_czi_channel_as_zyx(czi_path, neuron_channel_number)
img_nuclei = load_czi_channel_as_zyx(czi_path, dapi_channel_number)
img_cancer = load_czi_channel_as_zyx(czi_path, cancer_channel_number)

if img_neurons.shape != img_nuclei.shape or img_neurons.shape != img_cancer.shape:
    raise ValueError(
        f"Image shapes differ: neurons={img_neurons.shape}, "
        f"nuclei={img_nuclei.shape}, cancer={img_cancer.shape}")

if CROP_FOR_TESTING is not None:
    img_neurons = img_neurons[CROP_FOR_TESTING]
    img_nuclei = img_nuclei[CROP_FOR_TESTING]
    img_cancer = img_cancer[CROP_FOR_TESTING]
# %% DRAW ROIS
if DRAW_ROIS:
    projection = img_neurons.max(axis=0)
    projection_2 = img_nuclei.max(axis=0)
    projection_3 = img_cancer.max(axis=0)

    viewer = napari.Viewer()
    viewer.add_image(
        projection,
        name="Neuron max projection for ROI drawing",
        scale=VOXEL_SCALE_ZYX[1:])

    viewer.add_image(
        projection_2,
        name="Nuclei max projection for ROI drawing",
        scale=VOXEL_SCALE_ZYX[1:],
        blending="additive")

    viewer.add_image(projection_3, 
                     name="Cancer max projection for ROI drawing", 
                     scale=VOXEL_SCALE_ZYX[1:], 
                     blending="additive")

    shapes_layer = viewer.add_shapes(
        name="Draw ROIs here",
        ndim=2,
        shape_type="polygon",
        edge_width=2,
        face_color="transparent",
        blending="additive")

    print("Draw ROIs in napari. Close the napari window after the entire analysis is finished.\nROIs get saved in the next cell.")
    napari.run()

    # add a "waiter" until the napari window is closed, then save the ROIs:
    

    """ roi_labels_2d = rasterize_shapes_to_labelmask(
        shapes_layer,
        image_shape_yx=projection.shape,
        scale_yx=VOXEL_SCALE_ZYX[1:])

    tifffile.imwrite(roi_tif_path, roi_labels_2d.astype(np.uint16))
    print(f"Saved ROI label mask to:\n{roi_tif_path}") """
else:
    roi_labels_2d = tifffile.imread(roi_tif_path).astype(np.uint16)
    print(f"Loaded ROI label mask from:\n{roi_tif_path}")
# %% SAVE ROIS (IF ANY)
if DRAW_ROIS:
    roi_labels_2d = rasterize_shapes_to_labelmask(
        shapes_layer,
        image_shape_yx=projection.shape,
        scale_yx=VOXEL_SCALE_ZYX[1:])

    tifffile.imwrite(roi_tif_path, roi_labels_2d.astype(np.uint16))
    print(f"Saved ROI label mask to:\n{roi_tif_path}\nPlease leave the napari window open until the entire analysis is finished.")
# %% SEGMENT CANCER CHANNEL WITH OTSU
cancer_mask, cancer_labels, cancer_threshold, img_cancer_corrected = segment_cancer_channel(
    img_cancer,
    roi_labels_2d=roi_labels_2d,
    method=CANCER_SEGMENTATION_METHOD,
    percentile=CANCER_THRESHOLD_PERCENTILE,
    gaussian_sigma=CANCER_GAUSSIAN_SIGMA,
    background_sigma=CANCER_BACKGROUND_SIGMA,
    min_object_voxels=CANCER_MIN_OBJECT_VOXELS,
    min_hole_voxels=CANCER_MIN_HOLE_VOXELS,
    apply_closing=CANCER_APPLY_CLOSING)

print(f"\nSaving cancer mask to TIFF file...")
tifffile.imwrite(cancer_mask_out_path, cancer_labels.astype(np.uint32))
print(f"Saved cancer Otsu mask to:\n{cancer_mask_out_path}")

if OPEN_RESULTS:
    print("\nOpening results in napari...")
    if "viewer" not in globals() or viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(img_cancer, 
                     name="Cancer channel", 
                     scale=VOXEL_SCALE_ZYX, 
                     blending="additive", 
                     colormap="red",
                     channel_axis=None)

    viewer.add_image(img_cancer_corrected, 
                     name="Cancer channel background corrected", 
                     scale=VOXEL_SCALE_ZYX, 
                     blending="additive", 
                     colormap="yellow",
                     channel_axis=None)
    
    viewer.add_labels(cancer_labels, 
                      name="Cancer Otsu labels", 
                      blending="additive", 
                      scale=VOXEL_SCALE_ZYX)

    # turn off visibility of the 4 images from the ROI drawing step:
    viewer.layers["Neuron max projection for ROI drawing"].visible = False
    viewer.layers["Nuclei max projection for ROI drawing"].visible = False
    viewer.layers["Cancer max projection for ROI drawing"].visible = False
    viewer.layers["Draw ROIs here"].visible = False

    napari.run()
# %% RUN CELLPOSE PER ROI
if PROCESS_ROIS:
    roi_ids = np.unique(roi_labels_2d)
    roi_ids = roi_ids[roi_ids != 0]

    print(f"Found {len(roi_ids)} ROIs: {roi_ids}")

    model_neurons = models.CellposeModel(gpu=GPU, model_type="cyto3")
    model_nuclei = models.CellposeModel(gpu=GPU, model_type="nuclei")

    full_neuron_masks = np.zeros(img_neurons.shape, dtype=np.uint32)
    full_nuclei_masks = np.zeros(img_nuclei.shape, dtype=np.uint32)

    neuron_label_offset = 0
    nuclei_label_offset = 0

    all_rows = []

    for roi_id in roi_ids:
        print(f"\nProcessing ROI {roi_id} with Cellpose...")

        roi_mask_2d = roi_labels_2d == roi_id
        bbox = get_bbox_2d(roi_mask_2d)

        if bbox is None:
            print(f"Skipping ROI {roi_id}: empty ROI")
            continue

        y_slice, x_slice = bbox

        roi_mask_crop_2d = roi_mask_2d[y_slice, x_slice]

        neuron_crop = img_neurons[:, y_slice, x_slice].copy()
        nuclei_crop = img_nuclei[:, y_slice, x_slice].copy()

        neuron_crop[:, ~roi_mask_crop_2d] = 0
        nuclei_crop[:, ~roi_mask_crop_2d] = 0

        masks_neuron_roi, flows_neuron, styles_neuron = model_neurons.eval(
            neuron_crop,
            do_3D=True,
            z_axis=0,
            channel_axis=None,
            diameter=NEURON_DIAMETER)

        masks_nuclei_roi, flows_nuclei, styles_nuclei = model_nuclei.eval(
            nuclei_crop,
            do_3D=True,
            z_axis=0,
            channel_axis=None,
            diameter=NUCLEI_DIAMETER)

        masks_neuron_roi = masks_neuron_roi.astype(np.uint32)
        masks_nuclei_roi = masks_nuclei_roi.astype(np.uint32)

        masks_neuron_roi = relabel_with_offset(masks_neuron_roi, neuron_label_offset)
        masks_nuclei_roi = relabel_with_offset(masks_nuclei_roi, nuclei_label_offset)

        if masks_neuron_roi.max() > 0:
            neuron_label_offset = int(masks_neuron_roi.max())

        if masks_nuclei_roi.max() > 0:
            nuclei_label_offset = int(masks_nuclei_roi.max())

        full_neuron_masks[:, y_slice, x_slice] = np.maximum(
            full_neuron_masks[:, y_slice, x_slice],
            masks_neuron_roi)

        full_nuclei_masks[:, y_slice, x_slice] = np.maximum(
            full_nuclei_masks[:, y_slice, x_slice],
            masks_nuclei_roi)

        rows = analyze_colocalization(
            masks_neuron_roi,
            masks_nuclei_roi,
            roi_id=int(roi_id),
        )

        for row in rows:
            row["y_min"] = int(y_slice.start)
            row["y_max"] = int(y_slice.stop)
            row["x_min"] = int(x_slice.start)
            row["x_max"] = int(x_slice.stop)

        all_rows.extend(rows)


    # FILTER SMALL NEURON LABELS:
    print(f"\nFiltering neuron labels smaller than {MIN_NEURON_VOXELS} voxels...")
    full_neuron_masks = filter_labels_by_size(full_neuron_masks, MIN_NEURON_VOXELS)
    # Recompute detailed colocalization after filtering:
    all_rows = []
    for roi_id in roi_ids:
        roi_mask_2d = roi_labels_2d == roi_id
        bbox = get_bbox_2d(roi_mask_2d)

        if bbox is None:
            continue

        y_slice, x_slice = bbox

        neuron_roi = full_neuron_masks[:, y_slice, x_slice]
        nuclei_roi = full_nuclei_masks[:, y_slice, x_slice]

        rows = analyze_colocalization(neuron_roi, nuclei_roi, roi_id=int(roi_id))

        for row in rows:
            row["y_min"] = int(y_slice.start)
            row["y_max"] = int(y_slice.stop)
            row["x_min"] = int(x_slice.start)
            row["x_max"] = int(x_slice.stop)

        all_rows.extend(rows)
    print("...done.")

    df = pd.DataFrame(all_rows)
    
    # CREATE NEURON REGIONPROPS TABLE:
    print("\nCreating neuron regionprops table...")
    neuron_props = regionprops_table(full_neuron_masks, properties=("label", "area", "centroid"))
    df_neuron_props = pd.DataFrame(neuron_props)
    df_neuron_props = df_neuron_props.rename(columns={
        "label": "neuron_label",
        "area": "neuron_voxels_props",
        "centroid-0": "centroid_z",
        "centroid-1": "centroid_y",
        "centroid-2": "centroid_x",
    })
    df_neuron_props["neuron_label"] = df_neuron_props["neuron_label"].astype(int)
    print("...done.")
    
    
    
    # CREATE NEURON SUMMARY TABLE:
    print("\nCreating neuron summary table...")
    summary_rows = []
    for roi_id in np.unique(df["roi_id"]):
        df_roi = df[df["roi_id"] == roi_id]
        for neuron_label in np.unique(df_roi["neuron_label"]):
            df_neuron = df_roi[df_roi["neuron_label"] == neuron_label]

            neuron_voxels = int(df_neuron["neuron_voxels"].iloc[0])
            n_overlapping_nuclei = int(df_neuron["n_overlapping_nuclei"].max())

            best_idx = df_neuron["overlap_voxels"].idxmax()

            best_overlap_voxels = int(df_neuron.loc[best_idx, "overlap_voxels"])
            best_overlap_fraction = float(df_neuron.loc[best_idx, "overlap_fraction_of_neuron"])
            best_nucleus_label = df_neuron.loc[best_idx, "nucleus_label"]

            dapi_positive = (
                (n_overlapping_nuclei > 0)
                and (best_overlap_voxels >= MIN_OVERLAP_VOXELS)
                and (best_overlap_fraction >= COLOC_THRESHOLD)
            )

            summary_rows.append({
                "roi_id": int(roi_id),
                "neuron_label": int(neuron_label),
                "neuron_voxels": neuron_voxels,
                "dapi_positive": bool(dapi_positive),
                "n_overlapping_nuclei": n_overlapping_nuclei,
                "best_nucleus_label": int(best_nucleus_label) if not pd.isna(best_nucleus_label) else np.nan,
                "best_overlap_voxels": best_overlap_voxels,
                "best_overlap_fraction": best_overlap_fraction,
            })
    df_summary = pd.DataFrame(summary_rows)
    df_summary = df_summary.merge(df_neuron_props, on="neuron_label", how="left")
    # Optional consistency check:
    df_summary["neuron_voxels_delta"] = (df_summary["neuron_voxels"] - df_summary["neuron_voxels_props"])
    print("...done.")
    
    
    # CREATE ROI OVERVIEW TABLE:
    print("\nCreating ROI overview table...")
    overview_rows = []

    z_size_um, y_size_um, x_size_um = VOXEL_SCALE_ZYX
    pixel_area_um2 = y_size_um * x_size_um
    voxel_volume_um3 = z_size_um * y_size_um * x_size_um
    n_z = img_neurons.shape[0]

    for roi_id in np.unique(roi_labels_2d):
        if roi_id == 0:
            continue

        roi_mask_2d = roi_labels_2d == roi_id
        roi_area_px = int(roi_mask_2d.sum())
        roi_area_um2 = float(roi_area_px * pixel_area_um2)

        roi_volume_voxels = int(roi_area_px * n_z)
        roi_volume_um3 = float(roi_volume_voxels * voxel_volume_um3)

        roi_mask_3d = np.repeat(roi_mask_2d[np.newaxis, :, :], n_z, axis=0)

        neuron_labels_roi = np.unique(full_neuron_masks[roi_mask_3d])
        neuron_labels_roi = neuron_labels_roi[neuron_labels_roi != 0]

        nuclei_labels_roi = np.unique(full_nuclei_masks[roi_mask_3d])
        nuclei_labels_roi = nuclei_labels_roi[nuclei_labels_roi != 0]

        df_roi_summary = df_summary[df_summary["roi_id"] == roi_id]

        if "cancer_mask" in globals():
            cancer_mask_3d_roi = cancer_mask & roi_mask_3d
            cancer_volume_voxels = int(cancer_mask_3d_roi.sum())
            cancer_volume_um3 = float(cancer_volume_voxels * voxel_volume_um3)
            cancer_coverage_3d_percent = float(100 * cancer_volume_voxels / roi_volume_voxels) if roi_volume_voxels > 0 else np.nan

            cancer_projection_2d = cancer_mask.any(axis=0)
            cancer_area_px = int((cancer_projection_2d & roi_mask_2d).sum())
            cancer_area_um2 = float(cancer_area_px * pixel_area_um2)
            cancer_coverage_2d_percent = float(100 * cancer_area_px / roi_area_px) if roi_area_px > 0 else np.nan
        else:
            cancer_area_px = np.nan
            cancer_area_um2 = np.nan
            cancer_volume_voxels = np.nan
            cancer_volume_um3 = np.nan
            cancer_coverage_2d_percent = np.nan
            cancer_coverage_3d_percent = np.nan

        overview_rows.append({
            "roi_id": int(roi_id),
            "n_neurons": int(len(neuron_labels_roi)),
            "n_dapi_positive_neurons": int(df_roi_summary["dapi_positive"].sum()),
            "(n_dapi_nuclei)": int(len(nuclei_labels_roi)),
            "drawn_roi_area_px": roi_area_px,
            #"roi_area_um2": roi_area_um2,
            "roi_volume_voxels": roi_volume_voxels,
            "roi_volume_um3": roi_volume_um3,
            #"cancer_area_px_2d_projection": cancer_area_px,
            #"cancer_area_um2_2d_projection": cancer_area_um2,
            #"cancer_coverage_2d_percent": cancer_coverage_2d_percent,
            "cancer_volume_voxels_3d": cancer_volume_voxels,
            "cancer_volume_um3_3d": cancer_volume_um3,
            "cancer_coverage_3d_percent": cancer_coverage_3d_percent,
        })
    df_overview = pd.DataFrame(overview_rows)
    print("...done.")

    if len(df) > 0:
        df = df.sort_values(
            by=["roi_id", "neuron_label", "overlap_voxels"],
            ascending=[True, True, False],
        )

    df.to_csv(csv_out_path, index=False)
    excel_out_path = czi_path.with_name(czi_path.stem + "_cellpose_colocalization.xlsx")
    with pd.ExcelWriter(excel_out_path) as writer:
        df.to_excel(
            writer,
            sheet_name="detailed_overlaps",
            index=False)

        df_summary.to_excel(
            writer,
            sheet_name="neuron_summary",
            index=False)

        df_overview.to_excel(
            writer,
            sheet_name="roi_overview",
            index=False)
    print(f"Saved Excel analysis to:\n{excel_out_path}")

    print(f"\nSaving neuron and nuclei masks to TIFF files...")
    tifffile.imwrite(neuron_mask_out_path, full_neuron_masks.astype(np.uint32))
    tifffile.imwrite(nuclei_mask_out_path, full_nuclei_masks.astype(np.uint32))
    print(f"\nSaved colocalization CSV to:\n{csv_out_path}")
    print(f"Saved neuron masks to:\n{neuron_mask_out_path}")
    print(f"Saved nuclei masks to:\n{nuclei_mask_out_path}")
# %% OPEN RESULTS
# CREATE DAPI-POSITIVE NEURON MASK LAYER:
print("\nCreating DAPI-positive neuron mask layer...")
positive_neuron_labels = df_summary.loc[df_summary["dapi_positive"],"neuron_label"].astype(np.uint32).to_numpy()
max_label = int(full_neuron_masks.max())
positive_neuron_labels = positive_neuron_labels[positive_neuron_labels <= max_label]
label_lut = np.zeros(max_label + 1, dtype=np.uint32)
label_lut[positive_neuron_labels] = positive_neuron_labels
dapi_positive_neuron_masks = label_lut[full_neuron_masks]
print(f"N DAPI-positive neurons: {len(positive_neuron_labels)}")
print("...done. Now opening results in napari...")

if OPEN_RESULTS:
    #viewer = napari.Viewer()
    if "viewer" not in globals() or viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(
        img_neurons,
        name="Neurons",
        scale=VOXEL_SCALE_ZYX,
        blending="additive",
        colormap="magenta",
        channel_axis=None)

    viewer.add_image(
        img_nuclei,
        name="Nuclei / DAPI",
        scale=VOXEL_SCALE_ZYX,
        blending="additive",
        colormap="cyan",
        channel_axis=None)

    """ viewer.add_image(img_cancer, 
                     name="Cancer channel", 
                     scale=VOXEL_SCALE_ZYX, 
                     blending="additive", 
                     colormap="red",
                     channel_axis=None) """

    roi_labels_3d = np.repeat(
        roi_labels_2d[np.newaxis, :, :],
        img_neurons.shape[0],
        axis=0)

    viewer.add_labels(
        roi_labels_3d,
        name="ROIs",
        blending="additive",
        scale=VOXEL_SCALE_ZYX)
    
    roi_points_yx, roi_text_labels = get_roi_label_points(roi_labels_2d)

    viewer.add_points(
        roi_points_yx,
        name="ROI numbers",
        scale=VOXEL_SCALE_ZYX[1:],
        size=10,
        face_color="transparent",
        #edge_color="white",
        text={
            "string": roi_text_labels,
            "size": 14,
            "color": "white",
            "anchor": "center",
        },
    )
    
    """ viewer.add_image(img_cancer_corrected, 
                     name="Cancer channel background corrected", 
                     scale=VOXEL_SCALE_ZYX, 
                     blending="additive", 
                     colormap="red",
                     channel_axis=None)
    
    viewer.add_labels(cancer_labels, 
                      name="Cancer Otsu labels", 
                      blending="additive", 
                      scale=VOXEL_SCALE_ZYX) """

    # turn off visibility of the 4 images from the ROI drawing step:
    viewer.layers["Neuron max projection for ROI drawing"].visible = False
    viewer.layers["Nuclei max projection for ROI drawing"].visible = False
    viewer.layers["Cancer max projection for ROI drawing"].visible = False
    viewer.layers["Draw ROIs here"].visible = False


    if PROCESS_ROIS:
        viewer.add_labels(
            full_neuron_masks,
            name="Cellpose neuron masks",
            blending="additive",
            scale=VOXEL_SCALE_ZYX)
        viewer.add_labels(
            full_nuclei_masks,
            name="Cellpose nuclei masks",
            blending="additive",
            scale=VOXEL_SCALE_ZYX)
        viewer.add_labels(
            dapi_positive_neuron_masks,
            name="DAPI-positive neuron masks",
            blending="additive",
            scale=VOXEL_SCALE_ZYX)
    napari.run()
# %% END