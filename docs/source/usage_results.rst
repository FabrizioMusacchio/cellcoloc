Understanding exported results
==============================

CellColoc writes its final analysis outputs to the dataset-specific
``results/`` directory. The exact set of files depends on the workflow
variant, but in general you will find:

- exported label masks as TIFF files,
- at least one CSV table,
- and one Excel workbook that bundles the structured result tables.

This page explains the result tables in a workflow-independent way so the
individual tutorials can stay focused on the analysis logic itself.


Multi-channel colocalization workbook
-------------------------------------

For two-channel or three-channel colocalization workflows, the Excel workbook
contains several sheets. These can be grouped into:

- colocalization-metric sheets,
- channel object-property sheets,
- and per-ROI morphology summary sheets.


Colocalization metric sheets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``detailed_overlaps``
^^^^^^^^^^^^^^^^^^^^^

This is the most granular colocalization table.

Each row represents one concrete overlap event between:

- one segmented cell object from the cell channel,
- and one segmented object from the marker channel.

Typical uses:

- inspect exactly which objects overlapped,
- understand why a specific cell became positive or negative,
- debug over-segmentation or under-segmentation.

Typical columns include:

- ``roi_id``
- ``cell_label``
- ``marker_label``
- ``cell_voxels``
- ``marker_voxels``
- ``overlap_voxels``
- ``overlap_fraction_of_cell``
- ROI crop bounds such as ``y_min`` / ``y_max`` / ``x_min`` / ``x_max``


``cell_summary``
^^^^^^^^^^^^^^^^

This is the main per-cell results table for colocalization workflows.

Each row represents one segmented cell object from the cell channel. The table
aggregates all overlap events for that cell and stores the final
classification outcome.

Typical colocalization columns include:

- ``roi_id``:
  ROI label in which the cell was analyzed.
- ``cell_label``:
  label ID of the segmented cell object.
- ``cell_voxels``:
  direct voxel count of the segmented cell object.
- ``marker_positive``:
  final boolean positivity call for the marker channel.
- ``n_overlapping_markers``:
  how many distinct marker objects touched this cell.
- ``best_marker_label``:
  marker object with the strongest overlap.
- ``best_overlap_voxels``:
  voxel count of the strongest overlap event.
- ``best_overlap_fraction``:
  fraction of the cell occupied by the strongest marker overlap.

If a third analysis channel is configured with additional positivity
evaluation, the table may also contain:

- ``optional_region_positive``
- ``n_overlapping_optional_region_objects``
- ``best_optional_region_label``
- ``best_optional_region_overlap_voxels``
- ``best_optional_region_overlap_fraction``
- ``marker_and_optional_region_positive``

In current CellColoc versions, ``cell_summary`` always also contains the
cell-channel morphology metrics, so that this sheet combines:

- colocalization state,
- object size,
- centroid information,
- and shape descriptors.

These morphology columns are described below under
`Channel object-property sheets`_ because they follow the same metric logic as
the dedicated channel-property sheets.


``roi_coloc_overview``
^^^^^^^^^^^^^^^^^^^^^^

This is the most compact ROI-level table for colocalization workflows.

Each row represents one ROI and combines:

- ROI geometry,
- counts of segmented objects,
- counts of positive cells,
- and occupancy metrics for each segmented channel.

This is usually the best first sheet to inspect when comparing ROIs across one
dataset or across multiple files.

ROI identity and counts
^^^^^^^^^^^^^^^^^^^^^^^

- ``roi_id``:
  integer ROI label.
- ``n_cells``:
  number of segmented cell-channel objects in the ROI.
- ``n_marker_positive_cells``:
  number of cells classified as marker positive.
- ``n_marker_objects``:
  number of segmented marker-channel objects.

If a third channel participates in positivity analysis, you may additionally
see:

- ``n_optional_region_positive_cells``
- ``n_marker_and_optional_region_positive_cells``

ROI geometry
^^^^^^^^^^^^

- ``drawn_roi_area_px``:
  ROI area in pixels.
- ``drawn_roi_area_um2``:
  ROI area in square micrometers.
- ``roi_volume_voxels``:
  effective ROI volume in analyzed voxels.
- ``roi_volume_um3``:
  physical ROI volume in cubic micrometers.

Channel occupancy blocks
^^^^^^^^^^^^^^^^^^^^^^^^

For each segmented channel, CellColoc reports one occupancy block with a
channel-specific prefix:

- ``cell_occupancy_*``
- ``marker_occupancy_*``
- ``optional_region_occupancy_*`` if a third channel is analyzed

Each block contains:

- 2D projected occupied area in pixels and square micrometers,
- 2D projected coverage percentage,
- 3D occupied voxel volume and physical volume,
- 3D coverage percentage.


Channel object-property sheets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same multi-channel colocalization workbook also contains channel-specific
object-property sheets:

- ``marker_properties``
- ``3rd_channel_properties`` when a third channel was segmented

For colocalization workflows, the cell-channel object properties are not
written to a separate sheet. Instead, they are included directly in
``cell_summary``. The marker and optional third-channel objects use the same
metric logic, but are written to their own sheets.

Each row in these tables corresponds to one segmented object in the respective
channel.

Shared identifier columns
~~~~~~~~~~~~~~~~~~~~~~~~~

Typical identifier columns are:

- ``roi_id``
- one label column such as ``cell_label``, ``marker_label``, or
  ``optional_region_label``
- ``centroid_z``
- ``centroid_y``
- ``centroid_x``

2D metrics
~~~~~~~~~~

For effectively 2D analyses, or for z-projected workflows, CellColoc reports
2D object descriptors:

- ``*_area_px_2d``
- ``*_area_um2_2d``
- ``*_perimeter_px_2d``
- ``*_perimeter_um_2d``
- ``*_roundness_2d``
- ``*_eccentricity_2d``

Interpretation:

- ``area`` quantifies object size in the image plane,
- ``perimeter`` describes boundary length,
- ``roundness`` is a compactness measure,
- ``eccentricity`` captures elongation in 2D.

3D metrics
~~~~~~~~~~

For true 3D analyses, CellColoc reports 3D descriptors:

- ``*_volume_voxels_3d``
- ``*_volume_um3_3d``
- ``*_surface_area_um2_3d``
- ``*_sphericity_3d``
- ``*_ellipticity_3d``

Interpretation:

- ``volume`` measures object size in the stack,
- ``surface_area`` estimates the voxel-surface envelope,
- ``sphericity`` captures how sphere-like an object is,
- ``ellipticity`` reflects 3D elongation.


Per-ROI morphology summary sheets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same multi-channel colocalization workbook also contains ROI-level
morphology summary sheets that average object properties within each ROI:

- ``roi_cell_summary``
- ``roi_marker_summary``
- ``roi_3rd_channel_summary`` when a third channel was segmented

Each row corresponds to one ROI.

These tables typically contain:

- ``roi_id``
- one object-count column such as ``n_cells`` or ``n_marker_objects``
- many ``average_*`` columns

Examples:

- ``average_cell_area_px_2d``
- ``average_marker_roundness_2d``
- ``average_optional_region_volume_um3_3d``

These sheets are useful when you want to compare not only how many objects a
ROI contains, but also whether the objects in that ROI are on average:

- larger or smaller,
- more elongated or more compact,
- more spherical or less spherical.


Single-channel workbook
-----------------------

For dedicated single-channel workflows, the exported workbook uses a slightly
different structure:

- ``object_summary``
- ``voxel_plausibility_check``
- ``roi_overview``

``object_summary``
~~~~~~~~~~~~~~~~~~

One row per segmented object with the biologically relevant size and shape
descriptors.

``voxel_plausibility_check``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Technical cross-check table comparing:

- direct voxel counts from the label mask,
- and voxel counts derived through ``regionprops``.

This is mainly a diagnostic sheet and is usually not the primary downstream
analysis table.

``roi_overview``
~~~~~~~~~~~~~~~~

One row per ROI containing:

- object counts,
- occupancy metrics,
- and per-ROI averages of the object-shape descriptors.
