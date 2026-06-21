API Reference
=============

The API reference below is generated automatically from the public ``cellcoloc``
package and its docstrings.


Top-level package
-----------------

.. automodule:: cellcoloc
   :members:
   :undoc-members:
   :show-inheritance:


Public API overview
-------------------

.. currentmodule:: cellcoloc

.. autosummary::
   :toctree: generated
   :recursive:

   CellposeModelConfig
   ChannelConfig
   ColocalizationConfig
   DisplayNames
   OptionalRegionSegmentationConfig
   RuntimeConfig
   ResultsPaths
   LoadedImageChannels
   OptionalRegionSegmentationResult
   ColocalizationTables
   ColocalizationRunResult
   build_results_paths
   load_analysis_images
   save_roi_labels
   load_roi_labels
   try_load_roi_labels
   export_analysis_outputs
   analyze_existing_masks
   prepare_loaded_images_for_analysis
   prepare_runtime_environment
   get_runtime_cache_root
   create_full_image_roi_labels
   rasterize_shapes_to_labelmask
   create_roi_drawing_viewer
   save_roi_labels_from_shapes
   get_bbox_2d
   get_roi_label_points
   create_cellpose_model
   create_cellpose_models_for_channels
   evaluate_cellpose_model
   relabel_with_offset
   filter_labels_by_size
   get_cellpose_major_version
   get_available_cellpose_model_names
   segment_optional_region
   run_roi_cellpose_colocalization
   refine_run_result_from_cellpose_cache
   build_positive_cell_mask
   extract_label_masks_from_viewer
   show_optional_region_segmentation
   show_analysis_results

