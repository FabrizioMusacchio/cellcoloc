CellColoc Documentation
========================

.. figure:: _static/CellColoc_2.png
   :alt: CellColoc overview
   :align: center
   :figwidth: 70%

­

.. image:: https://badgen.net/badge/icon/GitHub%20repository?icon=github&label
   :target: https://github.com/FabrizioMusacchio/cellcoloc/
   :alt: GitHub Repository

.. image:: https://img.shields.io/github/v/release/FabrizioMusacchio/cellcoloc
   :alt: GitHub Release

.. image:: https://img.shields.io/pypi/v/cellcoloc.svg
   :target: https://pypi.org/project/cellcoloc/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/License-GPL%20v3-green.svg
   :target: https://cellcoloc.readthedocs.io/en/latest/overview.html#license
   :alt: GPLv3 License

.. .. image:: https://github.com/FabrizioMusacchio/cellcoloc/actions/workflows/cellcoloc_tests.yml/badge.svg
..    :alt: Tests

.. image:: https://img.shields.io/github/last-commit/FabrizioMusacchio/cellcoloc
   :target: https://github.com/FabrizioMusacchio/cellcoloc/commits/main/
   :alt: GitHub last commit

.. image:: https://img.shields.io/codecov/c/github/FabrizioMusacchio/cellcoloc?logo=codecov
   :target: https://codecov.io/gh/fabriziomusacchio/cellcoloc
   :alt: codecov

.. image:: https://img.shields.io/github/issues/FabrizioMusacchio/cellcoloc
   :target: https://github.com/FabrizioMusacchio/cellcoloc/issues
   :alt: GitHub Issues Open

.. image:: https://img.shields.io/github/issues-closed/FabrizioMusacchio/cellcoloc?color=53c92e
   :target: https://github.com/FabrizioMusacchio/cellcoloc/issues?q=is%3Aissue%20state%3Aclosed
   :alt: GitHub Issues Closed

.. image:: https://img.shields.io/github/issues-pr/FabrizioMusacchio/cellcoloc
   :target: https://github.com/FabrizioMusacchio/cellcoloc/pulls
   :alt: GitHub Issues or Pull Requests

.. image:: https://readthedocs.org/projects/cellcoloc/badge/?version=latest
   :target: https://cellcoloc.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/github/languages/code-size/fabriziomusacchio/cellcoloc
   :alt: GitHub code size in bytes

.. image:: https://img.shields.io/pypi/dm/cellcoloc?logo=pypy&label=PiPY%20downloads&color=blue
   :target: https://pypistats.org/packages/cellcoloc
   :alt: PyPI Downloads

.. .. image:: https://static.pepy.tech/personalized-badge/cellcoloc?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=PiPY+total+downloads
..    :target: https://pepy.tech/projects/cellcoloc
..    :alt: PyPI Total Downloads

.. image:: https://img.shields.io/badge/Example%20Datasets-10.5281%2Fzenodo.20788293-blue
   :target: https://doi.org/10.5281/zenodo.20788293
   :alt: CellColoc example datasets on Zenodo

.. image:: https://img.shields.io/badge/Zenodo%20Archive-10.5281%2Fzenodo.20787509-blue
   :target: https://doi.org/10.5281/zenodo.20787509
   :alt: Zenodo Archive



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
