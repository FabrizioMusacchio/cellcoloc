# How to Contribute

Thank you for your interest in contributing to OMIO. This project welcomes improvements to the code, documentation, tests, and overall usability. The goal of OMIO is to provide a robust, transparent, and reproducible interface for reading, standardizing, and converting microscopy image data, with a strong focus on OME-compatible metadata handling and downstream interoperability.

## Before you start
Please check the GitHub issue tracker to see whether your idea, bug report, or enhancement has already been discussed:

[https://github.com/FabrizioMusacchio/omio/issues](https://github.com/FabrizioMusacchio/omio/issues)

* If a related issue exists, comment there to indicate your interest or to add relevant technical details.
* If no issue exists, open a new one with a short description of:
  * what you would like to change or add
  * why it is useful in the context of OMIO
  * any thoughts on implementation, edge cases, or testing

For small fixes such as typos or minor documentation improvements, opening a pull request directly is fine.

## Development environment
OMIO requires **Python 3.12 or newer** and builds on standard scientific Python packages commonly used in microscopy workflows, including NumPy, tifffile, zarr, and related libraries for metadata handling.

A typical development setup using `conda` looks like this:

```sh
git clone https://github.com/FabrizioMusacchio/omio.git
cd omio

conda create -n omio-dev -c conda-forge python=3.12
conda activate omio-dev

pip install -e .
````

To install optional development dependencies such as testing and linting tools:

```sh
pip install -e ".[dev]"
```

## Making changes and opening pull requests
All code contributions should be submitted as pull requests (PRs) against the `main` branch of the repository.

A recommended workflow:

1. Create a new feature branch:

   ```sh
   git checkout -b feature/my-feature
   ```

2. Implement your changes. New functions or modules should include clear docstrings explaining:
   * their purpose
   * expected inputs and outputs
   * any assumptions or limitations
3. Add tests for new functionality or bug fixes where appropriate.
4. Push your branch and open a pull request that includes:
   * a concise and descriptive title
   * a brief explanation of what was changed and why
   * references to related issues (for example “Closes #12”)

Draft pull requests are welcome if you would like feedback during development.

## Commit conventions
Clear and consistent commit messages help keep the project history readable. Prefixes inspired by Conventional Commits are encouraged:

* `feat:` new functionality
* `fix:` bug fixes
* `docs:` documentation changes
* `refactor:` internal code restructuring without behavior changes
* `test:` adding or modifying tests
* `chore:` maintenance tasks or tooling updates

Example:
`fix: handle paginated TIFF files with mixed photometric interpretations`

## Testing
OMIO uses `pytest` for automated testing. To run the full test suite locally:

```sh
pytest
```

If you add new features or fix bugs, please extend the test suite accordingly.

Tests should remain small and self-contained. Large microscopy datasets should not be added to the repository. Whenever possible, use synthetic arrays or minimal example files generated during the test run.


## Notes for JOSS-related contributions  *(new)*
OMIO is developed with the requirements of the *Journal of Open Source Software (JOSS)* in mind. Contributions should therefore respect the following principles, which are routinely evaluated during JOSS review:

* **Reproducibility**
  Behavior should be deterministic given identical inputs and parameters. Any non-deterministic behavior must be explicitly documented.
* **Test coverage**
  New functionality should be accompanied by tests that fail without the change and pass with it. Tests should target observable behavior rather than internal implementation details.
* **Documentation consistency**
  Public-facing functions must be documented in a way that is consistent with their actual behavior. Silent assumptions or undocumented side effects are discouraged.
* **Minimal scope changes**
  Pull requests should focus on a well-defined change. Large refactors or conceptual redesigns should be discussed in an issue before implementation.
* **Explicit limitations**
  Known limitations or unsupported cases should be documented rather than implicitly ignored.

Following these guidelines helps ensure that OMIO remains reviewable, maintainable, and suitable for long-term archival publication.


## OME policy decisions and design constraints
OMIO makes a number of explicit policy decisions when reading and converting microscopy data to OME-compatible representations. These decisions are intentional and are meant to favor robustness and downstream interoperability over implicit heuristics.

Key principles include:

* **Canonical axis normalization**
  OMIO internally normalizes image data to the canonical OME axis order `TZCYX`. Missing axes may be inserted with length 1, but ambiguous or non-OME axis labels are not silently reinterpreted.
* **Single-series default behavior**
  For multi-series TIFF files, OMIO currently processes only the first series by default. This behavior is recorded in the output metadata and is considered a policy decision rather than a technical limitation.
* **Metadata preservation over inference**
  Existing OME-XML and ImageJ metadata are preserved wherever possible. OMIO avoids inventing or guessing metadata fields that are not present in the source file.
* **Explicit handling of unsupported metadata**
  Metadata fields that are detected but not yet supported are reported explicitly rather than silently ignored. This is intended to make limitations visible and reproducible.

Contributions that alter or extend these policy decisions should be discussed in an issue before implementation, as such changes may affect reproducibility, compatibility with downstream tools, or consistency with existing datasets.


## Reporting bugs
Please report bugs via the GitHub issue tracker:

[https://github.com/FabrizioMusacchio/omio/issues](https://github.com/FabrizioMusacchio/omio/issues)

Include the following information if possible:

* OMIO version (`pip show omio`)
* Python version
* Operating system
* Minimal steps or code snippet to reproduce the issue
* If applicable, a small synthetic or cropped example file illustrating the problem

## Requests for new file formats and reader extensions
In addition to direct code contributions via pull requests, users are encouraged to request support for additional microscopy file formats or format variants that are not yet covered by OMIO.

Such requests should be submitted via the GitHub issue tracker and include:

* a clear description of the file format or variant in question
* how the file differs from formats already supported by OMIO
* which part of the reader pipeline fails or behaves unexpectedly
* if available, relevant OMIO output such as warnings, parsed metadata, or axis interpretations

Support for new formats or variants can only be added if a **representative example file** is made available. This is essential to ensure correct parsing, reproducibility, and long-term test coverage.

Example files can be shared via:
* temporary download links (for example institutional web shares or cloud storage)
* publicly accessible repositories or archives
* other means that allow the developers to locally inspect and test the data

Without access to an example file, reader extensions are generally not feasible, as OMIO deliberately avoids speculative or heuristic-based format inference.

If sharing full datasets is not possible, users are encouraged to provide the smallest possible cropped or anonymized file that still reproduces the issue.


## License and contributions
By submitting a pull request, you agree that your contributions will be released under the project’s license as specified in the repository.

If you are unsure how to begin or would like to discuss a potential contribution, feel free to open an issue to start a conversation.

