# Cell Coloc: A Python package for Cellpose-based colocalization analysis in microscopy images

This repository contains a small reusable core pipeline in `cell_coloc` plus project-specific interactive scripts in `user_scripts`.

## Goal
The package is designed for Cellpose-based two-channel colocalization workflows:

- channel 1: larger cell compartment to segment, for example cytoskeleton or soma,
- channel 2: marker object defining positivity, for example DAPI-positive nuclei.

An optional third channel can be enabled to threshold a region mask and quantify its area or volume fraction inside each ROI, for example tumor infiltration.

## Current structure
- `cell_coloc`: reusable core functions for I/O, ROI handling, optional third-channel segmentation, Cellpose processing, table creation, export, and napari visualization.
- `user_scripts/annabell__czi_user_script.py`: interactive example script for the dataset in `example_data/czi_private_3D`.
- `prototype/annabell__czi.py`: original prototype kept as reference.

## Workflow
Run the user script cell by cell in the VS Code interactive window:

1. set paths and analysis parameters in the settings cell,
2. load the dataset,
3. draw ROIs in napari and save them,
4. optionally segment the third channel,
5. run the ROI-wise Cellpose analysis,
6. inspect tables and masks in napari.

All outputs are written automatically to a `results` subfolder next to the raw data file.

## Notes
- Cellpose model selection is configurable via `model_name_or_path`. You can use built-in model names such as `"cyto3"` or `"nuclei"` or provide a custom model path.
- The pipeline now fails explicitly if a requested Cellpose model name is not available in the installed Cellpose version, instead of silently switching to another model.
- The deprecated scikit-image calls from the prototype have been replaced with the current morphology API.
- The code structure is ready for a later packaging step, so turning this into an installable PyPI package should be straightforward.
