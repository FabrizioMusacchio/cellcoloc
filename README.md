# Cell Coloc: A Python package for Cellpose-based colocalization analysis in microscopy images

*Cell Coloc* is a Python workflow for (optional) ROI-based colocalization analysis in 3D microscopy images using Cellpose for segmentation. 

It is designed for experiments where you want to:

- segment a larger cellular structure in one channel, such as soma, cytoplasm, or a neuronal compartment,
- segment a marker-defining structure in a second channel, such as DAPI-positive nuclei,
- classify each segmented cell as marker-positive or marker-negative based on object overlap,
- optionally quantify a third channel as a thresholded region mask, for example tumor infiltration or lesion coverage.

The current repository is built around interactive analysis with napari and Cellpose. It is intended for researchers who want a transparent workflow they can inspect, tune, and adapt to their own microscopy datasets.

## What this project does
For each manually defined ROI, the pipeline:

1. loads the selected microscopy channels,
2. lets you draw 2D ROIs in napari on maximum-intensity projections,
3. applies each ROI across the full z-stack,
4. segments cells and marker objects independently with Cellpose,
5. measures overlap between cell objects and marker objects,
6. classifies cells as marker-positive or marker-negative,
7. exports tables and label masks to a standardized `results/` folder,
8. optionally segments a third channel and reports its 2D and 3D coverage inside each ROI.

This makes the workflow useful for questions such as:

- How many cells in each ROI are positive for a given marker?
- What fraction of segmented cells overlap with nuclei or another marker-defined object?
- How much of an ROI is occupied by an additional tissue compartment or pathology-related signal?

## Intended Use Case
The default analysis model is a generic two-channel colocalization problem:

- `cell channel`: the larger biological object you want to count and segment,
- `marker channel`: the smaller or more specific structure that defines positivity.

Example mappings include:

- neurons and DAPI-positive nuclei,
- soma and marker-positive puncta,
- cells and channel-specific reporter signal.

An optional third channel can be used for region-level quantification, for example:

- tumor infiltration,
- lesion area,
- dense tissue signal,
- any thresholdable compartment of interest.

## Repository Layout

- `cell_coloc/`
  Reusable core package containing image loading, ROI handling, Cellpose segmentation, colocalization logic, optional third-channel segmentation, export, and visualization helpers.
- `user_scripts/`
  Interactive scripts that configure a concrete dataset and call the reusable pipeline step by step.
- `example_data/`
  Example project data and example output files.

## Analysis workflow
The recommended way to run the project is to execute a user script cell by cell in the VS Code interactive window or in a Jupyter-like environment.

Typical workflow:

1. set the input file path,
2. define channel indices,
3. define voxel size in z, y, x,
4. choose Cellpose models and diameter settings,
5. set colocalization thresholds,
6. load the images,
7. draw ROIs in napari,
8. optionally segment the third channel,
9. run ROI-wise Cellpose analysis,
10. inspect result tables and masks in napari.

All outputs are written automatically to a `results/` subfolder next to the raw microscopy file.

## How ROIs work
ROIs are drawn manually as 2D polygons in napari on max projections of the analysis channels.

Important detail:

- ROIs are drawn in 2D,
- the ROI mask is then applied through the full z-stack for 3D analysis.

This design is useful when the biological region of interest is easiest to define in projection, but the cell and marker measurements should still be computed volumetrically.

## How marker positivity is defined
Cells are segmented from the cell channel. Marker objects are segmented separately from the marker channel.

A cell is classified as marker-positive only if all of the following are true:

1. it overlaps with at least one marker object,
2. the best overlap reaches at least `min_overlap_voxels`,
3. the best overlap fraction reaches at least `overlap_fraction_threshold`.

This makes the classification rule explicit and easy to tune for different staining patterns or object sizes.

## Optional third-channel analysis
If enabled, the pipeline can segment a third channel by thresholding and morphology cleanup.

Supported thresholding modes currently include:

- `otsu`
- `li`
- `percentile`

Optional preprocessing and cleanup steps include:

- Gaussian smoothing,
- optional background subtraction,
- morphological closing,
- removal of small objects,
- removal of small holes.

The resulting region mask is then quantified inside each ROI as:

- 2D projected area,
- 2D coverage percentage,
- 3D volume,
- 3D coverage percentage.

## Outputs
For each analyzed dataset, the pipeline writes a standardized set of outputs into `results/`.

### Label masks
- ROI label mask
- Cellpose cell mask
- Cellpose marker mask
- marker-positive cell mask
- optional third-channel region mask

### Tables
The pipeline produces three levels of tabular output:

- `detailed`
  One row per cell-marker overlap event.
- `summary`
  One row per cell, including positivity call and centroid information.
- `overview`
  One row per ROI, including cell counts, marker-positive cell counts, marker object counts, ROI size, and optional third-channel coverage metrics.

The detailed table is exported as CSV. All tables are also exported together in an Excel workbook.

## Example configuration parameters
A typical analysis script defines:

- channel mapping via `ChannelConfig`
- display labels via `DisplayNames`
- Cellpose settings via `CellposeModelConfig`
- positivity thresholds via `ColocalizationConfig`
- optional region segmentation settings via `OptionalRegionSegmentationConfig`
- runtime toggles via `RuntimeConfig`

This keeps project-specific settings separate from the reusable analysis code.

## Installation
The repository is already structured like a reusable package, but it is not yet shipped as a PyPI package. At the moment, the simplest setup is a local Python environment plus manual dependency installation.

Example environment setup:

```bash
conda create -n cell-coloc python=3.12 -y
conda activate cell-coloc
conda install -y ipykernel
pip install cellpose napari matplotlib pandas openpyxl scikit-image tifffile omio-microscopy appdirs

