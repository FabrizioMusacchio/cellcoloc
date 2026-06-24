Usage
=====

This section contains practical workflow tutorials based on the interactive
user scripts provided with CellColoc.

CellColoc is intended to be used through project-specific scripts that
configure real datasets and then call the reusable package cell by cell in a
VS Code interactive window or notebook-like environment.

.. toctree::
   :maxdepth: 2

   usage_example_datasets
   usage_2d_dapi_stained_nuclei
   usage_results
   usage_3d_microglia
   usage_three_channel_analysis
   usage_three_channel_zprojection
   usage_2d_single_channel_dapi_nuclei
   


Current usage topics
--------------------

The documentation currently covers topics such as:

- configuring channels and display names
- loading microscopy datasets
- whole-image analysis versus ROI-based analysis
- ROI drawing and ROI reuse
- choosing segmentation backends
- using z-cropping and z-projection
- refining `Cellpose <https://www.cellpose.org>`_ results post hoc
- recomputing tables after manual mask edits
- understanding exported result tables and masks
- adapting the provided user scripts for new projects


Recommended starting point
--------------------------

If you are new to CellColoc, start with the 2D tutorial first. It introduces
the interactive analysis model with the least amount of complexity and shows
how a complete run is structured from configuration to export.

The 3D tutorial then builds on the same ideas and adds z-aware features such
as anisotropy handling, disk-backed loading, refinement-time z-cropping, and
manual reanalysis of edited 3D label masks.

The interactive scripts in the repository's ``user_scripts/`` directory show
the same structure on real datasets and can be adapted directly for new
projects. They show how to:

- configure a project,
- run the pipeline cell by cell,
- inspect results in napari,
- and export reproducible outputs.
