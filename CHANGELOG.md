## CellColoc changelog

See here for a detailed list of changes made in each release of *CellColoc*.
Please, also refer to the Repository [Releases page](https://github.com/FabrizioMusacchio/cellcoloc/releases).

Each release is also archived on Zenodo for long-term preservation and citation purposes:

[![Zenodo Archive](https://img.shields.io/badge/Zenodo%20Archive-10.5281%2Fzenodo.20787509-blue)](https://doi.org/10.5281/zenodo.20787509)

## 🔜 Next version

### ✨ Features

- extend the multi-channel colocalization export with channel-wise morphology
  tables and per-ROI morphology summaries:
  - augment ``cell_summary`` with cell-channel size and shape metrics
  - add ``marker_properties`` for segmented marker objects
  - add ``3rd_channel_properties`` when an optional third channel is analyzed
  - rename the ROI overview export sheet to ``roi_coloc_overview``
  - add ``roi_cell_summary``, ``roi_marker_summary``, and optional
    ``roi_3rd_channel_summary`` sheets with per-ROI mean morphology metrics


## 🚀 CellColoc v0.0.4

June 23, 2026

This release expands **CellColoc** from a pure multi-channel colocalization
workflow into a broader interactive microscopy-analysis toolkit by adding a
dedicated single-channel mode, a first full set of usage tutorials, and
tutorial-derived notebook counterparts for the interactive example scripts.

### ✨ Features

- add a dedicated single-channel segmentation and counting workflow that can
  analyze one microscopy channel without any colocalization step while still
  reusing CellColoc's existing core capabilities:
  - Cellpose and threshold-based segmentation backends
  - ROI-based or whole-image analysis
  - prefiltering and postfiltering
  - global z-cropping and optional z projection
  - cached Cellpose refinement
  - manual napari relabeling and reanalysis
  - standardized mask and table export
- add a dedicated 2D DAPI nuclei demo script for the new single-channel
  workflow
- extend the single-channel object export with morphology metrics:
  - 2D area, perimeter, roundness, and eccentricity
  - 3D volume, voxel-surface area, sphericity, and ellipticity-like
    elongation
  - a separate voxel plausibility sheet in the Excel export
  - per-ROI averages of the new morphology metrics

### 📃 Changes

- allow ``VOXEL_SCALE_ZYX`` to be provided either as a full ``(Z, Y, X)``
  tuple or, for 2D workflows, as a shorter ``(Y, X)`` tuple that is expanded
  internally to ``(1.0, Y, X)``
- add the first full usage tutorials to the Read the Docs documentation:
  - a 2D tutorial based on the DAPI-stained nuclei example workflow
  - a 3D tutorial based on the microglia example workflow
  - a three-channel tutorial
  - a three-channel z-projection tutorial
  - a 2D single-channel nuclei tutorial
- generate notebook counterparts for the interactive example workflows from
  the tutorial structure itself, including local figure references inside the
  ``user_scripts`` folder
- expand the documentation with mathematical definitions of object-based
  colocalization and occupancy metrics
- improve the Read the Docs configuration so copy buttons are shown on all
  standard highlighted code blocks instead of only Python code snippets
- extend the 2D DAPI example user script with:
  - whole-image-as-single-ROI mode
  - automatic reuse of an existing saved ROI mask from the results directory
- add a dedicated three-channel 3D microglia demo script that demonstrates:
  - active segmentation of the third channel
  - separate visualization of cells positive for channel ``0+1``
  - separate visualization of cells positive for channel ``0+2``
  - separate visualization of cells positive for channel ``0+1+2``
- add a dedicated three-channel z-projection demo script that demonstrates:
  - global z projection before segmentation
  - projected three-channel analysis
  - projected positivity views for ``0+1``, ``0+2``, and ``0+1+2``
- extend cache-based Cellpose refinement so the optional third analysis channel
  can also be rebuilt from cached Cellpose outputs, including optional
  threshold changes and postfiltering
- keep manual reanalysis after napari label edits consistent with the active
  analysis z-bounds in the 3D workflows
- surface the new single-channel workflow explicitly in the README and the
  general documentation overview as a first-class CellColoc feature


---

## 🚀 CellColoc v0.0.3

June 21, 2026

This release adds the first project-wide archival and example-data publication
records on Zenodo for **CellColoc**.

This release provides:

- an official Zenodo archive for **CellColoc** that can now be used for
  software citation:
  - DOI: `10.5281/zenodo.20787509`
  - Citation: Musacchio, F. (2026). *CellColoc: A Python package for
    interactive segmentation-based colocalization analysis in microscopy
    images*. Zenodo. https://doi.org/10.5281/zenodo.20787509
- a dedicated Zenodo example-data record for **CellColoc**:
  - DOI: `10.5281/zenodo.20788293`
- updated release metadata to reflect the new citable software archive and
  externally hosted example dataset

---

## 🚀 CellColoc v0.0.2

June 21, 2026

This release adds the initial Read the Docs documentation structure for **CellColoc**.

This release provides:

- a first Sphinx / Read the Docs documentation scaffold under ``docs/``
- initial documentation pages for:
  - project overview
  - installation
  - usage landing page
  - API reference
  - changelog
- automatic API-reference structuring based on the public ``cellcoloc`` package
- a prepared usage section that will later be expanded with dedicated
  user-script walkthroughs

Notes:

- the detailed user-script usage pages are intentionally still pending and will
  follow in a later documentation update

---

## 🚀 CellColoc v0.0.1

June 21, 2026

First public main release of **CellColoc**.

This initial release provides:

- the reusable `cellcoloc` Python package for interactive, segmentation-based colocalization analysis in microscopy images
- stepwise user-script workflows for VS Code interactive window and notebook-like execution
- OMIO-based microscopy loading with `TZCYX` handling
- automatic 2D versus 3D detection from the raw z dimension
- voxel-size resolution from explicit user input or OMIO metadata, with fallback to `(1.0, 1.0, 1.0)` when necessary
- channel-wise segmentation method selection with support for:
  - `cellpose`
  - `otsu`
  - `li`
  - `percentile`
- optional ROI drawing in napari
- optional whole-image analysis as one single ROI
- optional reuse of previously saved ROI masks
- per-cell overlap analysis and marker-positivity classification
- standardized detailed, summary, and overview result tables
- standardized export into a `results/` subfolder next to the raw dataset
- occupancy metrics for every segmented channel
- optional third-channel segmentation and occupancy quantification
- optional third-channel cell-positivity analysis and double-positive reporting
- optional global z-cropping for internal analysis
- optional global z-projection using:
  - `max`
  - `mean`
  - `median`
  - `std`
  - `var`
- optional anisotropy handling for true 3D Cellpose runs
- optional `flow3d_smooth` support for Cellpose
- optional image prefiltering with:
  - `gaussian`
  - `median`
  - `laplacian_of_gaussian`
  - ordered prefilter chains
- optional label postfiltering with:
  - `min_intensity`
  - `local_contrast`
  - `bright_pixel_support`
  - ordered postfilter chains
- fast Cellpose cache-based refinement using stored network outputs
- optional manual napari mask editing followed by table recomputation
- reusable visualization helpers with selective layer refreshing in napari
- runtime fallback handling for cache and config directories when desktop libraries cannot write to default locations
- packaging metadata for installation via `pip`

Packaging notes:

- PyPI package name: `cellcoloc`
- import name: `cellcoloc`
- optional interactive extra: `cellcoloc[interactive]`
- optional tested Cellpose 3 extra: `cellcoloc[cellpose3]`

---
