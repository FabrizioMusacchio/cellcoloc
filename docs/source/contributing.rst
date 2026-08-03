Contributing and community guidelines
=====================================

*CellColoc* is an open source project that evolves through contributions from its users
and the broader microscopy community. Contributions range from bug reports and
documentation improvements to new analysis features, segmentation helpers,
workflow examples, and result-export improvements.

The goal of *CellColoc* is to provide a robust, explicit, and reproducible
interface for segmentation-based colocalization analysis in microscopy images.
Contributions are therefore evaluated not only by functionality, but also by
clarity, reproducibility, and long term maintainability.

How to contribute
-----------------

If you are interested in contributing to *CellColoc*, the recommended entry points are:

* reporting bugs or unexpected behavior
* suggesting improvements to the documentation or examples
* requesting new colocalization, segmentation, filtering, or quantification
  features
* submitting pull requests with code changes

Bug reports and feature requests should be submitted via the
`GitHub issue tracker <https://github.com/FabrizioMusacchio/cellcoloc/issues>`_.
For code changes and larger contributions, please open a pull request against
the main repository.

Contribution guidelines
-----------------------

The repository contains a dedicated contribution guide in the file
``CONTRIBUTING.md`` (`link <https://github.com/FabrizioMusacchio/cellcoloc?tab=contributing-ov-file>`_). 
It describes in more detail:

* how to set up a local development environment
* the preferred workflow for branching and pull requests
* conventions for commit messages and code style
* expectations regarding tests and documentation

Before opening a pull request, please make sure that:

* the code is formatted consistently with the existing code base
* existing tests pass locally, and new functionality is covered by tests where applicable
* public functions and modules are documented via docstrings
* user facing changes are reflected in the documentation pages

Requests for new analysis methods and workflow extensions
---------------------------------------------------------

In addition to direct code contributions via pull requests, users are encouraged
to request new analysis capabilities that are not yet covered by *CellColoc*.
Examples include additional colocalization rules, new object-positivity
criteria, segmentation backends, prefilters, postfilters, ROI summaries, or
exported object metrics.

Such requests should be submitted via the GitHub issue tracker and include:

* a clear description of the biological or image-analysis question
* the expected input data structure, for example 2D, 3D, z-projected, two-channel,
  three-channel, or single-channel analysis
* the desired segmentation, filtering, or colocalization behavior
* how the requested method should be reflected in exported tables or masks
* if available, a minimal script snippet, configuration block, screenshot, or
  current CellColoc output that illustrates the need

For new analysis methods, representative example data are extremely helpful.
They allow contributors to verify that the method behaves as expected, document
its intended use, and add meaningful tests. When sharing data is not possible,
please provide the smallest possible synthetic or cropped example that still
captures the relevant behavior.

Useful supporting material includes:

* temporary download links, for example institutional web shares or cloud storage
* publicly accessible repositories or archives
* small synthetic arrays or masks that reproduce the requested behavior
* expected output tables or manually curated reference masks

Code of conduct
---------------

All interactions in the *CellColoc* project are governed by a `Code of Conduct <https://github.com/FabrizioMusacchio/cellcoloc?tab=coc-ov-file>`_ based on
the `Contributor Covenant <https://www.contributor-covenant.org>`_. By
participating in the project, you agree to abide by these guidelines.

If you experience or observe behavior that violates the Code of Conduct, please
report it via email to the maintainer.

Where to start
----------------

If you are looking for a first contribution, the issue tracker may contain issues
labeled as suitable starting points, for example documentation improvements,
small refactorings, tests for existing behavior, or narrowly scoped additions to
analysis helpers.

You are also welcome to open an issue to discuss ideas for new features or
workflow extensions before starting an implementation.

Thank you for considering contributing to *CellColoc* 🙏
