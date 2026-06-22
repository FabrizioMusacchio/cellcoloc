Usage
=====

This section is the entry point for future usage guides and workflow examples.

CellColoc is intended to be used through project-specific interactive user
scripts that configure real datasets and then call the reusable package step
by step. Those user-facing walkthrough pages will be added later after the
example scripts have been finalized and cleaned up for documentation.

.. toctree::
   :maxdepth: 2

   usage_example_datasets


Planned usage topics
--------------------

The future usage section is expected to cover topics such as:

- configuring channels and display names
- loading microscopy datasets
- ROI drawing and ROI reuse
- choosing segmentation backends
- using z cropping and z projection
- refining `Cellpose <https://www.cellpose.org>`_ results post hoc
- recomputing tables after manual mask edits
- understanding exported result tables and masks
- adapting the provided user scripts for new projects


Current recommendation
----------------------

For now, the best usage examples are the interactive scripts in the repository's
``user_scripts/`` directory. They show how to:

- configure a project,
- run the pipeline cell by cell,
- inspect results in napari,
- and export reproducible outputs.


