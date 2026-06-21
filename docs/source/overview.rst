Overview
========

.. figure:: _static/CellColoc_2.png
   :alt: CellColoc overview
   :align: center
   :figwidth: 70%

­

CellColoc is a Python package for interactive, segmentation-based
colocalization analysis in microscopy images.

The package is designed for workflows in which a user wants to:

- segment a larger biological object in one channel,
- segment or threshold a marker-defining structure in a second channel,
- classify cells as marker-positive or marker-negative based on overlap,
- optionally include a third channel for occupancy quantification or optional
  third-marker positivity,
- inspect, refine, and export results in a reproducible way.


Motivation
----------

Many microscopy analysis scripts start as project-specific prototypes and then
grow by repeated copying and adaptation. This quickly leads to duplicated core
logic, inconsistent result formats, and hidden workflow drift across projects.

CellColoc addresses this by separating:

- reusable core analysis functions inside the package,
- project-specific configuration and execution logic inside interactive user
  scripts.

This keeps the analysis transparent and inspectable for researchers while still
making the underlying workflow reusable across datasets and projects.


Core design
-----------

CellColoc is built around a generic channel model:

- one primary ``cell`` channel,
- one primary ``marker`` channel,
- one optional third analysis channel.

Each analysis channel can use one of several segmentation backends:

- ``cellpose``
- ``otsu``
- ``li``
- ``percentile``

This means the package is not restricted to Cellpose-only workflows. Neural
network segmentation and threshold-based segmentation can be combined on a
channel-by-channel basis.


Main features
-------------

CellColoc currently provides:

- OMIO-based microscopy loading
- automatic 2D versus 3D detection
- optional voxel-size resolution from OMIO metadata
- optional ROI drawing in napari
- optional whole-image analysis as a single ROI
- optional reuse of saved ROI masks
- ROI-wise segmentation and overlap analysis
- occupancy metrics for every segmented channel
- optional third-channel cell-positivity analysis
- optional global z cropping
- optional global z projection
- optional prefilter chains and postfilter chains
- fast Cellpose cache-based refinement
- optional manual mask editing followed by table recomputation
- standardized table and mask export into a ``results/`` folder


Interactive workflow
--------------------

The intended usage model is interactive and notebook-like:

1. define dataset paths and channel assignments,
2. choose segmentation settings,
3. load the analysis channels,
4. optionally draw or reload ROIs,
5. run segmentation and colocalization,
6. inspect results in napari,
7. optionally refine thresholds or masks,
8. export the final results.

This makes CellColoc especially suitable for exploratory but still
reproducible microscopy workflows.


License
-------

CellColoc is distributed under the terms of the GNU General Public License v3.0
or later (GPL-3.0-or-later).


Citation
--------

If you use CellColoc in scientific work, please cite:

Musacchio, F. (2026). *CellColoc: A Python package for interactive
segmentation-based colocalization analysis in microscopy images*. Zenodo.
https://doi.org/10.5281/zenodo.20787509

.. raw:: html

   <hr>

For questions, suggestions or bug reports, please refer to the
`GitHub issue tracker <https://github.com/FabrizioMusacchio/cellcoloc/issues>`_ of 
the `CellColoc repository <https://github.com/FabrizioMusacchio/cellcoloc>`_ or contact the maintainer 
directly:

| **Fabrizio Musacchio**: `Email <mailto:fabrizio.musacchio@dzne.de>`_ | `GitHub <https://github.com/FabrizioMusacchio>`_ | `Website <https://www.fabriziomusacchio.com>`_

