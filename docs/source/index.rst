CellColoc Documentation
========================

.. figure:: _static/CellColoc_2.png
   :alt: CellColoc overview
   :align: center
   :figwidth: 100%

­

.. image:: https://img.shields.io/pypi/v/cellcoloc.svg
   :target: https://pypi.org/project/cellcoloc/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/License-GPL%20v3-green.svg
   :alt: GPLv3 License

.. image:: https://readthedocs.org/projects/cellcoloc/badge/?version=latest
   :target: https://cellcoloc.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

`CellColoc <https://github.com/FabrizioMusacchio/CellColoc>`_ is a Python package
for interactive, segmentation-based colocalization analysis in microscopy
images. It combines reusable core analysis logic with project-specific user
scripts that can be executed step by step in VS Code or notebook-like
environments.

CellColoc supports both neural-network segmentation with Cellpose and
classical threshold-based segmentation on a per-channel basis, including ROI-
wise analysis, occupancy quantification, optional third-channel logic, fast
post hoc refinement, and interactive napari inspection.

.. note::
   CellColoc is under active development. The workflow and public API are
   already usable, but additional examples, usage guides, and packaging polish
   will continue to evolve.


.. toctree::
   :maxdepth: 3
   :caption: Contents

   overview
   installation
   usage
   api
   changelog
   contributing

