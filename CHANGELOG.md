## CellColoc Changelog

This file documents notable user-facing releases of CellColoc.

Repository releases:
[https://github.com/FabrizioMusacchio/CellColoc/releases](https://github.com/FabrizioMusacchio/CellColoc/releases)

Zenodo archive:
[https://doi.org/10.5281/zenodo.18030883](https://doi.org/10.5281/zenodo.18030883)

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
- optional global z cropping for internal analysis
- optional global z projection using:
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
