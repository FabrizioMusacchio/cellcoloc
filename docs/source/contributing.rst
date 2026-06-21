Contributing and Community Guidelines
=====================================

*CellColoc* is an open source project that evolves through contributions from its users
and the broader microscopy community. Contributions range from bug reports and
documentation improvements to new reader implementations and extensions of the
I/O pipeline.

The goal of *CellColoc* is to provide a robust, explicit, and reproducible interface for
reading, normalizing, and converting microscopy image data into OME compliant
representations. Contributions are therefore evaluated not only by functionality,
but also by clarity, reproducibility, and long term maintainability.

How to Contribute
-----------------

If you are interested in contributing to *CellColoc*, the recommended entry points are:

* reporting bugs or unexpected behavior
* suggesting improvements to the documentation or examples
* requesting support for additional file formats or format variants
* submitting pull requests with code changes

Bug reports, feature requests, and format support requests should be submitted
via the `GitHub issue tracker <https://github.com/FabrizioMusacchio/CellColoc/issues>`_.
For code changes and larger contributions, please open a pull request against
the main repository.

Contribution Guidelines
-----------------------

The repository contains a dedicated contribution guide in the file
```CONTRIBUTING.md`` <https://github.com/FabrizioMusacchio/cellcoloc?tab=contributing-ov-file>`_. It describes in more detail:

* how to set up a local development environment
* the preferred workflow for branching and pull requests
* conventions for commit messages and code style
* expectations regarding tests and documentation

Before opening a pull request, please make sure that:

* the code is formatted consistently with the existing code base
* existing tests pass locally, and new functionality is covered by tests where applicable
* public functions and modules are documented via docstrings
* user facing changes are reflected in the documentation pages

Requests for New File Formats and Reader Extensions
---------------------------------------------------

In addition to direct code contributions via pull requests, users are encouraged
to request support for additional microscopy file formats or format variants that
are not yet covered by *CellColoc*.

Such requests should be submitted via the GitHub issue tracker and include:

* a clear description of the file format or variant in question
* how the file differs from formats already supported by *CellColoc*
* which part of the reader or conversion pipeline fails or behaves unexpectedly
* if available, relevant *CellColoc* output such as warnings, parsed metadata, or axis
  interpretations

Support for new file formats or format variants can only be added if a
**representative example file** is made available. This is essential to ensure
correct parsing, reproducibility, and long term test coverage.

Example files can be shared via:

* temporary download links, for example institutional web shares or cloud storage
* publicly accessible repositories or archives
* other means that allow the developers to locally inspect and test the data

Without access to an example file, reader extensions are generally not feasible,
as *CellColoc* deliberately avoids speculative or heuristic based format inference.

If sharing full datasets is not possible, users are encouraged to provide the
smallest possible cropped or anonymized file that still reproduces the issue.

Code of Conduct
---------------

All interactions in the *CellColoc* project are governed by a `Code of Conduct <https://github.com/FabrizioMusacchio/cellcoloc?tab=coc-ov-file>`_ based on
the `Contributor Covenant <https://www.contributor-covenant.org>`_. By
participating in the project, you agree to abide by these guidelines.

If you experience or observe behavior that violates the Code of Conduct, please
report it via email to the maintainer.

Where to Start?
----------------

If you are looking for a first contribution, the issue tracker may contain issues
labeled as suitable starting points, for example documentation improvements,
small refactorings, or reader extensions for narrowly defined format variants.

You are also welcome to open an issue to discuss ideas for new features or reader
extensions before starting an implementation.

Thank you for considering contributing to *CellColoc* 🙏