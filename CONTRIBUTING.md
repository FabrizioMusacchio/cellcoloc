# How to Contribute

Thank you for your interest in contributing to CellColoc. This project welcomes improvements to the code, documentation, tests, example workflows, and overall usability. The goal of CellColoc is to provide a robust, transparent, and reproducible interface for segmentation-based colocalization analysis in microscopy images, including ROI-based workflows, channel-wise segmentation, post hoc refinement, object measurements, and standardized result export.

## Before you start
Please check the GitHub issue tracker to see whether your idea, bug report, or enhancement has already been discussed:

[https://github.com/FabrizioMusacchio/cellcoloc/issues](https://github.com/FabrizioMusacchio/cellcoloc/issues)

* If a related issue exists, comment there to indicate your interest or to add relevant technical details.
* If no issue exists, open a new one with a short description of:
  * what you would like to change or add
  * why it is useful in the context of CellColoc
  * any thoughts on implementation, edge cases, or testing

For small fixes such as typos or minor documentation improvements, opening a pull request directly is fine.

## Development environment
CellColoc requires **Python 3.12** and builds on standard scientific Python packages commonly used in microscopy workflows, including Cellpose, OMIO, napari, NumPy, pandas, scikit-image, tifffile, and related libraries for image analysis and result export.

A typical development setup using `conda` looks like this:

```sh
git clone https://github.com/FabrizioMusacchio/cellcoloc.git
cd cellcoloc

conda create -n cellcoloc-dev -c conda-forge python=3.12
conda activate cellcoloc-dev

pip install -e .
```

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
`fix: preserve marker labels during post hoc reanalysis`

## Testing
CellColoc uses `pytest` for automated testing. To run the full test suite locally:

```sh
pytest
```

If you add new features or fix bugs, please extend the test suite accordingly.

Tests should remain small and self-contained. Large microscopy datasets should not be added to the repository. Whenever possible, use synthetic arrays, small label masks, or minimal generated image stacks that exercise the behavior under test.


## Notes for JOSS-related contributions
CellColoc is developed with the requirements of the *Journal of Open Source Software (JOSS)* in mind. Contributions should therefore respect the following principles, which are routinely evaluated during JOSS review:

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

Following these guidelines helps ensure that CellColoc remains reviewable, maintainable, and suitable for long-term archival publication.


## Analysis policy decisions and design constraints
CellColoc makes a number of explicit policy decisions when turning microscopy images and segmentation masks into colocalization results. These decisions are intentional and are meant to favor reproducibility, transparent interpretation, and downstream comparability over hidden heuristics.

Key principles include:

* **Object-based colocalization**
  CellColoc evaluates colocalization from segmented objects and overlap criteria, not from raw intensity correlation. Changes to positivity logic should therefore be explicit, documented, and reflected in exported tables.
* **Channel-wise segmentation**
  Each analysis channel can use its own segmentation backend and filter chain. Contributions should preserve this per-channel configurability instead of hard-coding assumptions for a specific experiment.
* **ROI-wise reproducibility**
  ROI definitions, whole-image mode, z-cropping, and z-projection affect downstream measurements. New features should keep these analysis boundaries visible and consistently applied across masks, tables, and visualizations.
* **Transparent result export**
  Exported CSV and Excel columns should have stable, descriptive names. New result metrics should be documented and, where possible, accompanied by tests using small synthetic data.
* **Explicit limitations**
  Known limitations, backend-specific behavior, and unsupported cases should be documented rather than implicitly ignored.

Contributions that alter or extend these policy decisions should be discussed in an issue before implementation, as such changes may affect reproducibility, compatibility with existing user scripts, or interpretation of previously exported results.


## Reporting bugs
Please report bugs via the GitHub issue tracker:

[https://github.com/FabrizioMusacchio/cellcoloc/issues](https://github.com/FabrizioMusacchio/cellcoloc/issues)

Include the following information if possible:

* CellColoc version (`pip show cellcoloc`)
* Python version
* Operating system
* Minimal steps or code snippet to reproduce the issue
* Relevant CellColoc configuration blocks, for example `CellposeModelConfig`, `ColocalizationConfig`, and `RuntimeConfig`
* If applicable, a small synthetic example, cropped image, label mask, screenshot, or exported table illustrating the problem

## Requests for new analysis methods and workflow extensions
In addition to direct code contributions via pull requests, users are encouraged to request new analysis capabilities that are not yet covered by CellColoc. Examples include additional colocalization rules, new object-positivity criteria, segmentation backends, prefilters, postfilters, ROI summaries, or exported object metrics.

Such requests should be submitted via the GitHub issue tracker and include:

* a clear description of the biological or image-analysis question
* the expected input data structure, for example 2D, 3D, z-projected, two-channel, three-channel, or single-channel analysis
* the desired segmentation, filtering, or colocalization behavior
* how the requested method should be reflected in exported tables or masks
* if available, a minimal script snippet, configuration block, screenshot, or current CellColoc output that illustrates the need

For new analysis methods, representative example data are extremely helpful. They allow contributors to verify that the method behaves as expected, document its intended use, and add meaningful tests. When sharing data is not possible, please provide the smallest possible synthetic or cropped example that still captures the relevant behavior.

Useful supporting material can be shared via:
* temporary download links (for example institutional web shares or cloud storage)
* publicly accessible repositories or archives
* small synthetic arrays or masks that reproduce the requested behavior
* expected output tables or manually curated reference masks

Please do not add large microscopy datasets directly to the repository. For larger examples, use a stable external archive or provide a cropped/anonymized subset that is sufficient for testing.


## License and contributions
By submitting a pull request, you agree that your contributions will be released under the project’s license as specified in the repository.

If you are unsure how to begin or would like to discuss a potential contribution, feel free to open an issue to start a conversation.
