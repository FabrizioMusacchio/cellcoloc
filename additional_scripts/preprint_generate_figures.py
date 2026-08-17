"""Generate figure panels for the CellColoc preprint.

author: Fabrizio Musacchio
date:   July/August 2026

"""
# %% IMPORTS
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu, pearsonr, shapiro, ttest_rel, wilcoxon
from cellcoloc import ChannelConfig, load_analysis_images
# %% PATHS AND CONSTANTS
ROOT_DIR = Path(__file__).resolve().parents[1]
PREPRINT_DIR = ROOT_DIR / "papers" / "preprint"
FIGURES_DIR = PREPRINT_DIR / "figures"

SYNTHETIC_GAUSSIAN_DIR = ROOT_DIR / "example_data" / "synthetic_benchmark_data"
SYNTHETIC_SHARP_DIR = ROOT_DIR / "example_data" / "synthetic_benchmark_data_sharp"
SYNTHETIC_SHARP_CELLPOSE_DIR = SYNTHETIC_SHARP_DIR / "cellpose_variant_stacks"

MICROGLIA_FULL_DIR = ROOT_DIR / "example_data" / "microglia_3D_full"
MICROGLIA_MANIFEST_PATH = MICROGLIA_FULL_DIR / "microglia_3d_full_batch_manifest.csv"

CM_TO_INCH = 1.0 / 2.54
PANEL_DPI = 300
INSTANCE_MATCH_IOU_THRESHOLD = 0.50

GT_COLOR = "#17324d"
PRED_COLOR = "#d95f02"
CTX_COLOR = "#2f6b8a"
CA1_COLOR = "#cc7a00"
DEFAULT_SPINES = {"top": False, "right": False, "left": True, "bottom": True}
CTX_VEHICLE_COLOR = "#9cc4d6"
CTX_TAMOX_COLOR = CTX_COLOR
CA1_VEHICLE_COLOR = "#f0bc7b"
CA1_TAMOX_COLOR = CA1_COLOR
MICROGLIA_MORPHOLOGY_EFFECT_METRICS = [
    ("mean_cell_area_um2_2d", "area", "$\\mu$m$^2$"),
    ("mean_cell_roundness_2d", "roundness", ""),
    ("mean_cell_eccentricity_2d", "eccentricity", "")]

MICROGLIA_CELL_CMAP     = LinearSegmentedColormap.from_list("cell_red", ["#000000", "#ff3b30"])
MICROGLIA_MARKER_CMAP   = LinearSegmentedColormap.from_list("marker_cyan", ["#000000", "#00f0ff"])
MICROGLIA_OPTIONAL_CMAP = LinearSegmentedColormap.from_list("optional_yellow", ["#000000", "#ffd400"])

MICROGLIA_CHANNEL_CONFIG = ChannelConfig(
    cell_channel=0,
    marker_channel=1,
    optional_region_channel=2)
MICROGLIA_MOUSE_ID_ALIASES = {
    "22453": "22463"}
MICROGLIA_TREATMENT_BY_MOUSE_ID = {
    "22461": "Tamoxifen",
    "22462": "Tamoxifen",
    "22463": "Vehicle",
    "22464": "Vehicle",
    "22485": "Tamoxifen",
    "22486": "Tamoxifen",
    "22487": "Vehicle",
    "22488": "Vehicle"}
MICROGLIA_TREATMENT_XTICK_LABELS = {
    "Vehicle": "veh.",
    "Tamoxifen": "tam."}
# %% CENTRAL PANEL CONTROLS
@dataclass(frozen=True)
class PanelConfig:
    output_subdir: str
    output_name: str
    figsize_cm: tuple[float, float]
    title: str
    xlabel: str | None = None
    ylabel: str | None = None
    xlim: tuple[float | None, float | None] | None = None
    ylim: tuple[float | None, float | None] | None = None
    xticks: list[float] | None = None
    yticks: list[float] | None = None
    legend_show: bool = False
    legend_loc: str = "best"
    spines: dict[str, bool] = field(default_factory=lambda: DEFAULT_SPINES.copy())
    xrotation: float = 0.0
    ytick_length: float = 3.0
    show_scalebar: bool = False
    show_scalebar_unit: bool = False
    scalebar_position: str = "lower left"
    scalebar_length_microns: float | None = None
    scalebar_label: str | None = None
    transparent: bool | None = None
    cmap: str | None = None
    grid_axis: str | None = "y"
    title_pad: float = 6.0

def image_panel(
    output_subdir: str,
    output_name: str,
    title: str,
    *,
    figsize_cm: tuple[float, float] = (6.5, 6.5),
    cmap: str | None = None,
    show_scalebar: bool = False,
    show_scalebar_unit: bool = False,
    scalebar_position: str = "lower left",
    scalebar_length_microns: float | None = None,
    scalebar_label: str | None = None,
) -> PanelConfig:
    return PanelConfig(
        output_subdir=output_subdir,
        output_name=output_name,
        figsize_cm=figsize_cm,
        title=title,
        cmap=cmap,
        show_scalebar=show_scalebar,
        show_scalebar_unit=show_scalebar_unit,
        scalebar_position=scalebar_position,
        scalebar_length_microns=scalebar_length_microns,
        scalebar_label=scalebar_label,
        grid_axis=None)

def plot_panel(
    output_subdir: str,
    output_name: str,
    title: str,
    *,
    figsize_cm: tuple[float, float],
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    yticks: list[float] | None = None,
    legend_show: bool = False,
    legend_loc: str = "best",
    spines: dict[str, bool] | None = None,
    xrotation: float = 0.0,
    ytick_length: float = 3.0,
    grid_axis: str | None = "y",
) -> PanelConfig:
    return PanelConfig(
        output_subdir=output_subdir,
        output_name=output_name,
        figsize_cm=figsize_cm,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        yticks=yticks,
        legend_show=legend_show,
        legend_loc=legend_loc,
        spines=DEFAULT_SPINES.copy() if spines is None else spines.copy(),
        xrotation=xrotation,
        ytick_length=ytick_length,
        grid_axis=grid_axis)

def clone_panel_set(panel_set: dict[str, PanelConfig], output_subdir: str) -> dict[str, PanelConfig]:
    return {key: replace(panel, output_subdir=output_subdir)
            for key, panel in panel_set.items()}

SYNTHETIC_PANELS: dict[str, PanelConfig] = {
    "raw_channel0": image_panel("figure2", "panel_a.png", "Channel 0", figsize_cm=(5.4, 5.1), cmap="gray"),
    "raw_channel1": image_panel("figure2", "panel_b.png", "Channel 1", figsize_cm=(5.4, 5.1), cmap="gray"),
    "raw_channel2": image_panel("figure2", "panel_c.png", "Channel 2", figsize_cm=(5.4, 5.1), cmap="gray"),
    "raw_composite": image_panel("figure2", "panel_d.png", "Composite", figsize_cm=(5.4, 5.1)),
    "mask_channel0": image_panel("figure2", "panel_e.png", "CellColoc cell mask", figsize_cm=(5.4, 5.1)),
    "mask_channel1": image_panel("figure2", "panel_f.png", "CellColoc marker mask", figsize_cm=(5.4, 5.1)),
    "mask_region": image_panel("figure2", "panel_g.png", "CellColoc region mask", figsize_cm=(5.4, 5.1)),
    "mask_positive": image_panel("figure2", "panel_h.png", "CellColoc positive-cell mask", figsize_cm=(5.4, 5.1)),
    "counts_channel0": plot_panel(
        "figure2",
        "panel_i.pdf",
        "channel 0 counts",
        figsize_cm=(3.6, 4.4),
        ylabel="detected objects",
        ylim=(0, 35),
        xlim=(-0.4, 1.4),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "counts_channel1": plot_panel(
        "figure2",
        "panel_j.pdf",
        "channel 1 counts",
        figsize_cm=(3.6, 4.4),
        ylabel="detected objects",
        ylim=(0, 35),
        xlim=(-0.4, 1.4),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "counts_positive": plot_panel(
        "figure2",
        "panel_k.pdf",
        "colocalization counts",
        figsize_cm=(3.6, 4.4),
        ylabel="positive channel 0 cells",
        ylim=(0, 35),
        xlim=(-0.4, 1.4),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "coverage_channel2": plot_panel(
        "figure2",
        "panel_l.pdf",
        "channel 2\noccupancy",
        figsize_cm=(3.8, 4.4),
        ylabel="coverage [%]",
        ylim=(25, 35),
        xlim=(-0.4, 1.4),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "detection_metrics": plot_panel(
        "figure2",
        "panel_m.pdf",
        "instance detection metrics",
        figsize_cm=(5.2, 4.4),
        ylabel="score",
        ylim=(0.0, 1.35),
        legend_show=False,
        legend_loc="best",
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "error_summary": plot_panel(
        "figure2",
        "panel_n.pdf",
        "per-stack absolute\nerror summary",
        figsize_cm=(4.5, 5.3),
        ylabel="MAE",
        ylim=(0.0, 2.5),
        xrotation=25,
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "areas_matched": plot_panel(
        "figure2",
        "panel_p.pdf",
        "matched areas",
        figsize_cm=(5.6, 5.1),
        ylabel="predicted area (pixels)",
        xlim=(0, 700),
        ylim=(0, 700),
        xlabel="GT area (pixels)",
        grid_axis="both",
    ),
    "roundness_matched": plot_panel(
        "figure2",
        "panel_r.pdf",
        "matched roundness",
        figsize_cm=(5.6, 5.1),
        ylabel="predicted roundness",
        xlabel="GT roundness",
        xlim=(0.88, 1.12),
        ylim=(0.88, 1.12),
        grid_axis="both",
    ),
    "eccentricity_matched": plot_panel(
        "figure2",
        "panel_s.pdf",
        "matched eccentricity",
        figsize_cm=(5.6, 5.1),
        ylabel="predicted eccentricity",
        xlabel="GT eccentricity",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        grid_axis="both",
    ),
    "overlap_fraction_matched": plot_panel(
        "figure2",
        "panel_t.pdf",
        "matched overlap fraction",
        figsize_cm=(5.6, 5.1),
        ylabel="predicted overlap fraction",
        xlabel="GT overlap fraction",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
        grid_axis="both",
    ),
    "overlap_fraction_distribution": plot_panel(
        "figure2",
        "panel_u.pdf",
        "GT vs predicted\noverlap fraction",
        figsize_cm=(4.8, 5.3),
        ylabel="overlap fraction",
        ylim=(-0.02, 1.02),
        xlim=(-0.4, 1.4),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    )}

SYNTHETIC_SUPPLEMENT_PANELS: dict[str, PanelConfig] = clone_panel_set(SYNTHETIC_PANELS, "figureS1")
SYNTHETIC_GAUSSIAN_CELLPOSE_PANELS: dict[str, PanelConfig] = clone_panel_set(SYNTHETIC_PANELS, "figure_2_var")
SYNTHETIC_CELLPOSE_SUPPLEMENT_PANELS: dict[str, PanelConfig] = clone_panel_set(SYNTHETIC_PANELS, "figure_2_var2")

MICROGLIA_PANELS: dict[str, PanelConfig] = {
    "ctx_channel0": image_panel("figure4", "panel_a.png", "CTX channel 0", 
                                figsize_cm=(13.8, 13.8),
                                cmap=MICROGLIA_CELL_CMAP,
                                show_scalebar=False,
                                show_scalebar_unit=True,
                                scalebar_length_microns=50,
                                scalebar_position="lower right",
                                ),
    "ctx_channel1": image_panel("figure4", "panel_b.png", "CTX channel 1", 
                                figsize_cm=(13.8, 13.8),
                                cmap=MICROGLIA_MARKER_CMAP,
                                ),
    "ctx_channel2": image_panel("figure4", "panel_r.png", "CTX channel 2", 
                                figsize_cm=(13.8, 13.8),
                                cmap=MICROGLIA_OPTIONAL_CMAP,
                                ),
    "ctx_overlay": image_panel("figure4", "panel_c.png", "CTX overlay", 
                               figsize_cm=(13.8, 13.8),
                                show_scalebar=False,
                                show_scalebar_unit=False,
                                scalebar_length_microns=50,
                                scalebar_position="lower right",),
    "ca1_channel0": image_panel("figure4", "panel_d.png", "CA1 channel 0", figsize_cm=(13.8, 13.8), cmap=MICROGLIA_CELL_CMAP),
    "ca1_channel1": image_panel("figure4", "panel_e.png", "CA1 channel 1", figsize_cm=(13.8, 13.8), cmap=MICROGLIA_MARKER_CMAP),
    "ca1_channel2": image_panel("figure4", "panel_s.png", "CA1 channel 2", figsize_cm=(13.8, 13.8), cmap=MICROGLIA_OPTIONAL_CMAP),
    "ca1_overlay": image_panel("figure4", "panel_f.png", "CA1 overlay", figsize_cm=(13.8, 13.8)),
    "ctx_cell_mask": image_panel("figure4", "panel_g.png", "CTX cell mask", figsize_cm=(13.8, 13.8)),
    "ctx_marker_mask": image_panel("figure4", "panel_h.png", "CTX marker mask", figsize_cm=(13.8, 13.8)),
    "ctx_positive_mask": image_panel("figure4", "panel_i.png", "CTX positive-cell mask", figsize_cm=(13.8, 13.8)),
    "ca1_cell_mask": image_panel("figure4", "panel_j.png", "CA1 cell mask", figsize_cm=(13.8, 13.8)),
    "ca1_marker_mask": image_panel("figure4", "panel_k.png", "CA1 marker mask", figsize_cm=(13.8, 13.8)),
    "ca1_positive_mask": image_panel("figure4", "panel_l.png", "CA1 positive-cell mask", figsize_cm=(13.8, 13.8)),
    "n_cells": plot_panel(
        "figure4",
        "panel_m.pdf",
        "tdTomato-positive\ncell density",
        figsize_cm=(4.1, 4.9), #(5.0, 5)
        ylabel="cells / mm$^3$",
        ylim=(0, 21000),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "n_iba1_positive_cells": plot_panel(
        "figure4",
        "panel_n.pdf",
        "Iba1-positive\ntdTomato cell density",
        figsize_cm=(4.1, 4.9),
        ylabel="cells / mm$^3$",
        ylim=(0, 21000),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "iba1_positive_fraction": plot_panel(
        "figure4",
        "panel_o.pdf",
        "Iba1-positive\nfraction",
        figsize_cm=(3.8, 4.9),
        ylabel="fraction",
        ylim=(0.0, 1.15),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "mean_cell_area": plot_panel(
        "figure4",
        "panel_p.pdf",
        "mean cell area",
        figsize_cm=(3.8, 4.9),
        ylabel="area ($\\mu$m$^2$)",
        ylim=(0, None),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "mean_cell_roundness": plot_panel(
        "figure4",
        "panel_q.pdf",
        "mean cell roundness",
        figsize_cm=(3.8, 4.9),
        ylabel="roundness",
        ylim=(0.0, 1.2),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "mean_cell_eccentricity": plot_panel(
        "figure4",
        "panel_t.pdf",
        "mean cell eccentricity",
        figsize_cm=(3.8, 4.9),
        ylabel="eccentricity",
        ylim=(0.0, 1.05),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "mean_cell_brightness": plot_panel(
        "figure4",
        "panel_u.pdf",
        "average cell\nbrightness",
        figsize_cm=(3.8, 4.9),
        ylabel="mean intensity (a.u.)",
        ylim=(0, None),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "morphology_effect_raw": plot_panel(
        "figure4",
        "panel_v.pdf",
        "regional morphology effects",
        figsize_cm=(8.0, 4.9),
        xlabel="raw paired difference (CA1 - CTX)",
        spines={"top": False, "right": False, "left": False, "bottom": True},
        grid_axis="x",
    ),
    "morphology_effect_standardized": plot_panel(
        "figure4",
        "panel_w.pdf",
        "standardized morphology effects",
        figsize_cm=(7.0, 4.9),
        xlabel="paired difference / SD$_{diff}$ (CA1 - CTX)",
        xlim=(-3.0, 3.0),
        spines={"top": False, "right": False, "left": False, "bottom": True},
        grid_axis="x",
    ),
    "qc_density_positive_fraction": plot_panel(
        "figure4",
        "panel_x.pdf",
        "density vs marker-positive fraction",
        figsize_cm=(8.0, 4.9),
        xlabel="tdTomato-positive cells / mm$^3$",
        ylabel="Iba1-positive fraction",
        ylim=(0.0, 1.08),
        legend_show=True,
        legend_loc="best",
        spines={"top": False, "right": False, "left": True, "bottom": True},
    ),
    "cell_size_distribution_violin": plot_panel(
        "figure4",
        "panel_y.pdf",
        "tdTomato object-size QC",
        figsize_cm=(8.0, 4.9),
        ylabel="cell area (pixels)",
        ylim=(0, None),
        spines={"top": False, "right": False, "left": True, "bottom": True},
    ),
    "cell_size_distribution_strip": plot_panel(
        "figure4",
        "panel_z.pdf",
        "tdTomato object-size QC",
        figsize_cm=(8.0, 4.9),
        ylabel="cell area (pixels)",
        ylim=(0, None),
        spines={"top": False, "right": False, "left": True, "bottom": True},
    ),
    "cell_size_distribution_histogram": plot_panel(
        "figure4",
        "panel_aa.pdf",
        "tdTomato object-size QC",
        figsize_cm=(8.0, 4.9),
        xlabel="cell area (pixels)",
        ylabel="objects",
        xlim=(0, None),
        legend_show=True,
        legend_loc="best",
        spines={"top": False, "right": False, "left": True, "bottom": True},
    ),
    "area_roundness_scatter": plot_panel(
        "figure4",
        "panel_ab.pdf",
        "object morphology space",
        figsize_cm=(8.0, 4.9),
        xlabel="cell area (pixels)",
        ylabel="roundness",
        ylim=(0.0, 1.2),
        legend_show=True,
        legend_loc="best",
        spines={"top": False, "right": False, "left": True, "bottom": True},
    ),
    "count_median_area_scatter": plot_panel(
        "figure4",
        "panel_ac.pdf",
        "count vs median object size",
        figsize_cm=(8.0, 4.9),
        xlabel="tdTomato-positive cells / mm$^3$",
        ylabel="median cell area (pixels)",
        legend_show=True,
        legend_loc="best",
        spines={"top": False, "right": False, "left": True, "bottom": True},
    ),
    "iba1_overlap_fraction_histogram": plot_panel(
        "figure4",
        "panel_ad.pdf",
        "Iba1 overlap per tdTomato cell",
        figsize_cm=(8.0, 4.9),
        xlabel="best Iba1 overlap fraction",
        ylabel="tdTomato cells",
        xlim=(0.0, 1.02),
        legend_show=True,
        legend_loc="best",
        spines={"top": False, "right": False, "left": True, "bottom": True},
    ),
    "iba1_overlap_fraction_distribution": plot_panel(
        "figure4",
        "panel_ae.pdf",
        "Iba1 overlap decision margin",
        figsize_cm=(8.0, 4.9),
        ylabel="best Iba1 overlap fraction",
        ylim=(0.0, 1.05),
        spines={"top": False, "right": False, "left": True, "bottom": True},
    )}

MICROGLIA_ZOOM_PANELS: dict[str, PanelConfig] = {
    key: replace(panel, output_subdir="figure4_zoom")
    for key, panel in MICROGLIA_PANELS.items()
    if key.startswith(("ctx_", "ca1_"))}

MICROGLIA_TREATMENT_PANELS: dict[str, PanelConfig] = {
    "n_cells": plot_panel(
        "figure4_treatment",
        "panel_e.pdf",
        "tdTomato-positive\ncell density",
        figsize_cm=(6.2, 5.2),
        ylabel="cells / mm$^3$",
        ylim=(0, 21000),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "n_iba1_positive_cells": plot_panel(
        "figure4_treatment",
        "panel_f.pdf",
        "Iba1-positive\ntdTomato cell density",
        figsize_cm=(6.2, 5.2),
        ylabel="cells / mm$^3$",
        ylim=(0, 21000),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "iba1_positive_fraction": plot_panel(
        "figure4_treatment",
        "panel_g.pdf",
        "Iba1-positive\nfraction",
        figsize_cm=(5.8, 5.2),
        ylabel="fraction",
        ylim=(0.0, 1.15),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "mean_cell_area": plot_panel(
        "figure4_treatment",
        "panel_h.pdf",
        "mean cell area",
        figsize_cm=(5.8, 5.2),
        ylabel="area ($\\mu$m$^2$)",
        ylim=(0, None),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "mean_cell_roundness": plot_panel(
        "figure4_treatment",
        "panel_i.pdf",
        "mean cell roundness",
        figsize_cm=(5.8, 5.2),
        ylabel="roundness",
        ylim=(0.0, 1.2),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "mean_cell_eccentricity": plot_panel(
        "figure4_treatment",
        "panel_j.pdf",
        "mean cell eccentricity",
        figsize_cm=(5.8, 5.2),
        ylabel="eccentricity",
        ylim=(0.0, 1.05),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    )}

MICROGLIA_TREATMENT_GLOBAL_PANELS: dict[str, PanelConfig] = {
    "n_cells": plot_panel(
        "figure4_treatment_global",
        "panel_a.pdf",
        "tdTomato-positive\ncell density",
        figsize_cm=(4.1, 4.9),  # (4.9, 5.2)
        ylabel="cells / mm$^3$",
        ylim=(0, 21000),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "n_iba1_positive_cells": plot_panel(
        "figure4_treatment_global",
        "panel_b.pdf",
        "Iba1-positive\ntdTomato cell density",
        figsize_cm=(4.1, 4.9),
        ylabel="cells / mm$^3$",
        ylim=(0, 21000),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "iba1_positive_fraction": plot_panel(
        "figure4_treatment_global",
        "panel_c.pdf",
        "Iba1-positive\nfraction",
        figsize_cm=(3.8, 4.9),
        ylabel="fraction",
        ylim=(0.0, 1.15),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "mean_cell_area": plot_panel(
        "figure4_treatment_global",
        "panel_d.pdf",
        "mean cell area",
        figsize_cm=(4.1, 4.9),
        ylabel="area ($\\mu$m$^2$)",
        ylim=(0, None),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "mean_cell_roundness": plot_panel(
        "figure4_treatment_global",
        "panel_e.pdf",
        "mean cell roundness",
        figsize_cm=(4.1, 4.9),
        ylabel="roundness",
        ylim=(0.0, 1.2),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "mean_cell_eccentricity": plot_panel(
        "figure4_treatment_global",
        "panel_f.pdf",
        "mean cell eccentricity",
        figsize_cm=(4.1, 4.9),
        ylabel="eccentricity",
        ylim=(0.0, 1.05),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    ),
    "mean_cell_brightness": plot_panel(
        "figure4_treatment_global",
        "panel_g.pdf",
        "average cell\nbrightness",
        figsize_cm=(3.6, 4.9),
        ylabel="mean intensity (a.u.)",
        ylim=(0, None),
        spines={"top": False, "right": False, "left": False, "bottom": True},
    )}
# %% DATA CONTAINERS
@dataclass(slots=True)
class SyntheticEvaluation:
    stack_table: pd.DataFrame
    detection_long: pd.DataFrame
    gt_channel0_objects: pd.DataFrame
    pred_channel0_objects: pd.DataFrame
    matched_channel0: pd.DataFrame


@dataclass(frozen=True, slots=True)
class SyntheticBenchmarkPaths:
    data_dir: Path
    gt_dir: Path
    gt_label_dir: Path
    results_dir: Path


@dataclass(frozen=True, slots=True)
class MicrogliaFigureViewConfig:
    ctx_stack_name: str
    ca1_stack_name: str
    projection_method: str | None = "max"
    z_slice_index: int | None = None
    zoom_crop_size_px: int = 500
    include_optional_channel2_in_overlay: bool = True
    image_loading_mode: str = "memap"
    apply_holm_correction: bool = False


# Edit these two filenames to choose which CTX and CA1 stacks are loaded for
# the representative raw-image and mask panels in Figure 4 / figure4_zoom.
MICROGLIA_VIEW_CONFIG = MicrogliaFigureViewConfig(
    ctx_stack_name="250703_ID22462_ctx_tdTom_DAPI_IBA1_20x_1.czi",
    ca1_stack_name="250703_ID22462_CA1_tdTom_DAPI_IBA1_20x_2.czi",
    projection_method="max",
    z_slice_index=None,
    zoom_crop_size_px=500,
    include_optional_channel2_in_overlay=False,  # switch to turn on/off the optional channel 2 in the overlay panels
    image_loading_mode="memap",
    apply_holm_correction=False)

def make_synthetic_paths(data_dir: Path) -> SyntheticBenchmarkPaths:
    return SyntheticBenchmarkPaths(
        data_dir=data_dir,
        gt_dir=data_dir / "ground_truth",
        gt_label_dir=data_dir / "ground_truth" / "labels",
        results_dir=data_dir / "results")

SYNTHETIC_GAUSSIAN_PATHS = make_synthetic_paths(SYNTHETIC_GAUSSIAN_DIR)
SYNTHETIC_SHARP_PATHS = make_synthetic_paths(SYNTHETIC_SHARP_DIR)
SYNTHETIC_GAUSSIAN_CELLPOSE_PATHS = SyntheticBenchmarkPaths(
    data_dir=SYNTHETIC_GAUSSIAN_DIR / "cellpose_variant_stacks",
    gt_dir=SYNTHETIC_GAUSSIAN_DIR / "ground_truth",
    gt_label_dir=SYNTHETIC_GAUSSIAN_DIR / "ground_truth" / "labels",
    results_dir=SYNTHETIC_GAUSSIAN_DIR / "cellpose_variant_stacks" / "results")
SYNTHETIC_SHARP_CELLPOSE_PATHS = SyntheticBenchmarkPaths(
    data_dir=SYNTHETIC_SHARP_CELLPOSE_DIR,
    gt_dir=SYNTHETIC_SHARP_DIR / "ground_truth",
    gt_label_dir=SYNTHETIC_SHARP_DIR / "ground_truth" / "labels",
    results_dir=SYNTHETIC_SHARP_CELLPOSE_DIR / "results")
# %% GENERIC HELPERS
def cm_to_inch(value_cm: float) -> float:
    return value_cm * CM_TO_INCH

def ensure_output_dir(relative_subdir: str) -> Path:
    output_dir = FIGURES_DIR / relative_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def output_path_for(config: PanelConfig) -> Path:
    return ensure_output_dir(config.output_subdir) / config.output_name

def create_panel_figure(config: PanelConfig) -> tuple[plt.Figure, plt.Axes]:
    width_cm, height_cm = config.figsize_cm
    fig, ax = plt.subplots(figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)), dpi=PANEL_DPI)
    return fig, ax

def finalize_panel(fig: plt.Figure, output_path: Path, config: PanelConfig) -> None:
    fig.tight_layout(pad=0.45)
    transparent = config.transparent if config.transparent is not None else output_path.suffix.lower() == ".pdf"
    fig.savefig(output_path, bbox_inches="tight", transparent=transparent)
    plt.close(fig)

def apply_limits(ax: plt.Axes, config: PanelConfig) -> None:
    if config.xlim is not None:
        ax.set_xlim(*config.xlim)
    if config.ylim is not None:
        lower, upper = config.ylim
        if lower is not None or upper is not None:
            ax.set_ylim(lower, upper)

def apply_axes_controls(ax: plt.Axes, config: PanelConfig) -> None:
    if config.xlabel is not None:
        ax.set_xlabel(config.xlabel)
    if config.ylabel is not None:
        ax.set_ylabel(config.ylabel)
    ax.set_title(config.title, fontsize=10, pad=config.title_pad)
    apply_limits(ax, config)
    if config.xticks is not None:
        ax.set_xticks(config.xticks)
    if config.yticks is not None:
        ax.set_yticks(config.yticks)
    for spine_name, visible in config.spines.items():
        ax.spines[spine_name].set_visible(visible)
    ax.tick_params(axis="y", length=config.ytick_length)
    if config.xrotation:
        for label in ax.get_xticklabels():
            label.set_rotation(config.xrotation)
            label.set_ha("right")
    if config.grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(True, axis=config.grid_axis, alpha=0.25, linewidth=0.6)

def apply_legend_controls(ax: plt.Axes, config: PanelConfig) -> None:
    legend = ax.get_legend()
    if not config.legend_show:
        if legend is not None:
            legend.remove()
        return
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if config.legend_loc == "outer":
        ax.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        return
    ax.legend(handles, labels, frameon=False, loc=config.legend_loc)

def format_scalebar_microns_label(length_microns: float) -> str:
    if float(length_microns).is_integer():
        return f"{int(length_microns)} µm"
    return f"{length_microns:g} µm"

def add_scalebar_if_requested(
    ax: plt.Axes,
    image: np.ndarray,
    config: PanelConfig,
    *,
    microns_per_pixel: float | None = None,
) -> None:
    if not config.show_scalebar or config.scalebar_length_microns is None:
        return
    if microns_per_pixel is None or microns_per_pixel <= 0:
        raise ValueError(
            "A positive `microns_per_pixel` value is required when drawing a scalebar "
            "from `scalebar_length_microns`."
        )
    scalebar_length_px = float(config.scalebar_length_microns) / float(microns_per_pixel)
    height, width = image.shape[:2]
    margin_x = width * 0.06
    margin_y = height * 0.08
    line_y = height - margin_y
    if config.scalebar_position == "lower right":
        x0 = width - margin_x - scalebar_length_px
    else:
        x0 = margin_x
    ax.plot([x0, x0 + scalebar_length_px], [line_y, line_y], color="white", linewidth=3.0)
    if config.show_scalebar_unit:
        label = config.scalebar_label or format_scalebar_microns_label(float(config.scalebar_length_microns))
        ax.text(
            x0 + scalebar_length_px / 2.0,
            line_y - height * 0.04,
            label,
            color="white",
            ha="center",
            va="top",
            fontsize=8.5)

def build_stack_color_map(stack_ids: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    unique_ids = sorted(set(stack_ids))
    return {stack_id: cmap(index % 20) for index, stack_id in enumerate(unique_ids)}

def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(pearsonr(x, y)[0])

def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    n_tests = len(p_values)
    for rank, original_index in enumerate(order):
        multiplier = n_tests - rank
        adjusted_value = min(1.0, multiplier * p_values[original_index])
        running_max = max(running_max, adjusted_value)
        adjusted[original_index] = running_max
    return adjusted.tolist()
# %% IMAGE AND MASK HELPERS
def squeeze_to_2d(array: np.ndarray) -> np.ndarray:
    squeezed = np.asarray(array)
    while squeezed.ndim > 2:
        squeezed = squeezed[0]
    return squeezed

def read_first_plane(path: Path) -> np.ndarray:
    return squeeze_to_2d(tifffile.imread(path))

def synthetic_result_stem(stack_id: str) -> str:
    return f"{stack_id}.ome"

def load_synthetic_stack_image(paths: SyntheticBenchmarkPaths, stack_id: str) -> np.ndarray:
    image = tifffile.imread(paths.data_dir / f"{stack_id}.ome.tif")
    image = np.asarray(image)
    while image.ndim > 3:
        image = image[0]
    return image.astype(np.float32)

def normalize_image_for_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [1.0, 99.5])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)

def labels_to_rgb(label_image: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_image)
    if labels.dtype.kind == "b":
        labels = labels.astype(np.uint8)
    rgb = np.zeros(labels.shape + (3,), dtype=np.float32)
    positive_labels = [int(label) for label in np.unique(labels) if label != 0]
    cmap = plt.get_cmap("tab20")
    for index, label in enumerate(positive_labels):
        rgb[labels == label] = cmap(index % 20)[:3]
    return rgb

def choose_synthetic_example_stack(gt_summary: pd.DataFrame) -> str:
    target = float(gt_summary["n_channel0_cells_gt"].median())
    distances = (gt_summary["n_channel0_cells_gt"] - target).abs()
    return str(gt_summary.loc[distances.idxmin(), "stack_id"])

def save_image_panel(
    image: np.ndarray,
    config: PanelConfig,
    output_path: Path,
    *,
    cmap: str | None = None,
    microns_per_pixel: float | None = None,
) -> None:
    fig, ax = create_panel_figure(config)
    active_cmap = cmap if cmap is not None else config.cmap
    ax.imshow(image, cmap=active_cmap, interpolation="nearest")
    ax.set_title(config.title, fontsize=10, pad=config.title_pad)
    add_scalebar_if_requested(ax, np.asarray(image), config, microns_per_pixel=microns_per_pixel)
    ax.axis("off")
    finalize_panel(fig, output_path, config)

def save_label_mask_panel(mask: np.ndarray, config: PanelConfig, output_path: Path) -> None:
    save_image_panel(labels_to_rgb(mask), config, output_path)

def save_label_mask_panel_from_path(source_path: Path, config: PanelConfig, output_path: Path) -> None:
    save_label_mask_panel(read_first_plane(source_path), config, output_path)

def save_preview_panel_image(source_path: Path, config: PanelConfig, output_path: Path) -> None:
    image = plt.imread(source_path)
    save_image_panel(image, config, output_path)

def project_scalar_volume(
    image_zyx: np.ndarray,
    *,
    projection_method: str | None,
    z_slice_index: int | None,
) -> np.ndarray:
    image = np.asarray(image_zyx, dtype=np.float32)
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D scalar image, received shape {image.shape}.")
    if z_slice_index is not None:
        index = int(np.clip(z_slice_index, 0, image.shape[0] - 1))
        return image[index]
    if projection_method is None:
        return image[0]

    method = projection_method.strip().lower()
    if method == "max":
        return np.max(image, axis=0)
    if method == "mean":
        return np.mean(image, axis=0)
    if method == "median":
        return np.median(image, axis=0)
    raise ValueError(f"Unsupported microglia projection method: {projection_method!r}.")


def project_label_volume(
    label_image_zyx: np.ndarray,
    *,
    projection_method: str | None,
    z_slice_index: int | None,
) -> np.ndarray:
    labels = np.asarray(label_image_zyx)
    if labels.ndim == 2:
        return labels
    if labels.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D label image, received shape {labels.shape}.")
    if z_slice_index is not None:
        index = int(np.clip(z_slice_index, 0, labels.shape[0] - 1))
        return labels[index]
    if projection_method is None:
        return labels[0]
    return np.max(labels, axis=0)

def center_crop_image(image: np.ndarray, crop_size_px: int) -> np.ndarray:
    if crop_size_px <= 0:
        raise ValueError("`crop_size_px` must be positive.")
    array = np.asarray(image)
    height, width = array.shape[:2]
    crop_height = min(int(crop_size_px), height)
    crop_width = min(int(crop_size_px), width)
    y0 = max((height - crop_height) // 2, 0)
    x0 = max((width - crop_width) // 2, 0)
    y1 = y0 + crop_height
    x1 = x0 + crop_width
    return array[y0:y1, x0:x1, ...]

def choose_microglia_row_by_stack_name(manifest: pd.DataFrame, stack_name: str) -> pd.Series:
    subset = manifest[manifest["stack_name"] == stack_name]
    if subset.empty:
        available = ", ".join(sorted(manifest["stack_name"].astype(str).tolist())[:8])
        raise KeyError(
            f"Microglia stack {stack_name!r} was not found in the manifest. "
            f"Example available entries: {available}"
        )
    return subset.iloc[0]

def load_microglia_channel_views(
    source_path: Path,
    view_config: MicrogliaFigureViewConfig,
) -> tuple[dict[str, np.ndarray], float | None]:
    loaded_images = load_analysis_images(
        source_path=source_path,
        channel_config=MICROGLIA_CHANNEL_CONFIG,
        voxel_scale_zyx=None,
        crop_for_testing=None,
        image_loading_mode=view_config.image_loading_mode)
    channel0 = normalize_image_for_display(
        project_scalar_volume(
            loaded_images.cell_image,
            projection_method=view_config.projection_method,
            z_slice_index=view_config.z_slice_index))
    channel1 = normalize_image_for_display(
        project_scalar_volume(
            loaded_images.marker_image,
            projection_method=view_config.projection_method,
            z_slice_index=view_config.z_slice_index))
    if loaded_images.optional_region_image is not None:
        optional_region = normalize_image_for_display(
            project_scalar_volume(
                loaded_images.optional_region_image,
                projection_method=view_config.projection_method,
                z_slice_index=view_config.z_slice_index))
    else:
        optional_region = np.zeros_like(channel0)

    optional_weight = 0.7 if view_config.include_optional_channel2_in_overlay else 0.0
    overlay = np.zeros((*channel0.shape, 3), dtype=np.float32)
    overlay[..., 0] = np.clip(channel0 + optional_region * optional_weight, 0.0, 1.0)
    overlay[..., 1] = np.clip(channel1 + optional_region * optional_weight, 0.0, 1.0)
    overlay[..., 2] = np.clip(channel1, 0.0, 1.0)
    microns_per_pixel = float(loaded_images.voxel_scale_zyx[2]) if loaded_images.voxel_scale_zyx is not None else None
    return (
        {
            "channel0": channel0,
            "channel1": channel1,
            "channel2": optional_region,
            "overlay": overlay,
        }, microns_per_pixel,)

def load_microglia_mask_views(
    results_dir: Path,
    stack_stem: str,
    view_config: MicrogliaFigureViewConfig,
) -> dict[str, np.ndarray]:
    return {
        "cell_mask": project_label_volume(
            tifffile.imread(results_dir / f"{stack_stem}_cell_masks.tif"),
            projection_method=view_config.projection_method,
            z_slice_index=view_config.z_slice_index,
        ),
        "marker_mask": project_label_volume(
            tifffile.imread(results_dir / f"{stack_stem}_marker_masks.tif"),
            projection_method=view_config.projection_method,
            z_slice_index=view_config.z_slice_index,
        ),
        "positive_mask": project_label_volume(
            tifffile.imread(results_dir / f"{stack_stem}_positive_cell_masks.tif"),
            projection_method=view_config.projection_method,
            z_slice_index=view_config.z_slice_index,
        )}
# %% SYNTHETIC BENCHMARK EVALUATION HELPERS
def compute_iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray) -> tuple[list[int], list[int], np.ndarray]:
    gt_ids = [int(label) for label in np.unique(gt_labels) if label != 0]
    pred_ids = [int(label) for label in np.unique(pred_labels) if label != 0]
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
    for i, gt_label in enumerate(gt_ids):
        gt_mask = gt_labels == gt_label
        for j, pred_label in enumerate(pred_ids):
            pred_mask = pred_labels == pred_label
            intersection = int(np.count_nonzero(gt_mask & pred_mask))
            if intersection == 0:
                continue
            union = int(np.count_nonzero(gt_mask | pred_mask))
            matrix[i, j] = intersection / union
    return gt_ids, pred_ids, matrix

def match_instances(gt_labels: np.ndarray, pred_labels: np.ndarray, iou_threshold: float) -> dict[str, object]:
    gt_ids, pred_ids, iou = compute_iou_matrix(gt_labels, pred_labels)
    accepted_matches: list[dict[str, float | int]] = []
    if iou.size > 0:
        row_indices, col_indices = linear_sum_assignment(1.0 - iou)
        for row_index, col_index in zip(row_indices, col_indices):
            current_iou = float(iou[row_index, col_index])
            if current_iou < iou_threshold:
                continue
            accepted_matches.append(
                {
                    "gt_label": int(gt_ids[row_index]),
                    "pred_label": int(pred_ids[col_index]),
                    "iou": current_iou,
                })

    tp = len(accepted_matches)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else np.nan)
    return {
        "matches": accepted_matches,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_gt": len(gt_ids),
        "n_pred": len(pred_ids)}

def load_synthetic_evaluation(paths: SyntheticBenchmarkPaths) -> SyntheticEvaluation:
    gt_summary = pd.read_csv(paths.gt_dir / "synthetic_benchmark_ground_truth_summary.csv")
    gt_objects = pd.read_csv(paths.gt_dir / "synthetic_benchmark_ground_truth_objects.csv")
    gt_colocalization = pd.read_csv(paths.gt_dir / "synthetic_benchmark_ground_truth_colocalization.csv")

    stack_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    gt_channel0_rows: list[pd.DataFrame] = []
    pred_channel0_rows: list[pd.DataFrame] = []
    matched_channel0_rows: list[dict[str, object]] = []

    for gt_row in gt_summary.to_dict(orient="records"):
        stack_id = str(gt_row["stack_id"])
        result_stem = synthetic_result_stem(stack_id)

        overview = pd.read_csv(paths.results_dir / f"{result_stem}_roi_overview.csv").iloc[0]
        cell_summary = pd.read_csv(paths.results_dir / f"{result_stem}_cell_summary.csv")

        gt_channel0_labels = tifffile.imread(paths.gt_label_dir / f"{stack_id}_channel0_labels.tif")
        gt_channel1_labels = tifffile.imread(paths.gt_label_dir / f"{stack_id}_channel1_labels.tif")
        pred_channel0_labels = tifffile.imread(paths.results_dir / f"{result_stem}_cell_masks.tif")[0]
        pred_channel1_labels = tifffile.imread(paths.results_dir / f"{result_stem}_marker_masks.tif")[0]

        match_ch0 = match_instances(gt_channel0_labels, pred_channel0_labels, INSTANCE_MATCH_IOU_THRESHOLD)
        match_ch1 = match_instances(gt_channel1_labels, pred_channel1_labels, INSTANCE_MATCH_IOU_THRESHOLD)

        detection_rows.extend(
            [
                {
                    "stack_id": stack_id,
                    "channel": "Channel 0",
                    "precision": match_ch0["precision"],
                    "recall": match_ch0["recall"],
                    "f1": match_ch0["f1"],
                },
                {
                    "stack_id": stack_id,
                    "channel": "Channel 1",
                    "precision": match_ch1["precision"],
                    "recall": match_ch1["recall"],
                    "f1": match_ch1["f1"],
                },
            ]
        )

        gt_channel0_object_rows = gt_objects.query("stack_id == @stack_id and channel == 'channel0'").copy()
        gt_channel0_colocalization_rows = (
            gt_colocalization.query("stack_id == @stack_id")
            .loc[:, ["channel0_label", "best_overlap_fraction"]]
            .rename(
                columns={
                    "channel0_label": "label",
                    "best_overlap_fraction": "overlap_fraction",
                }
            )
        )
        gt_channel0_object_rows = gt_channel0_object_rows.merge(
            gt_channel0_colocalization_rows,
            on="label",
            how="left",
        )
        gt_channel0_rows.append(gt_channel0_object_rows)

        pred_channel0_object_rows = cell_summary.loc[
            :,
            [
                "cell_label",
                "cell_area_px_2d",
                "cell_roundness_2d",
                "cell_eccentricity_2d",
                "best_overlap_fraction",
            ],
        ].copy()
        pred_channel0_object_rows = pred_channel0_object_rows.rename(
            columns={
                "cell_label": "label",
                "cell_area_px_2d": "area_px_2d",
                "cell_roundness_2d": "roundness_2d",
                "cell_eccentricity_2d": "eccentricity_2d",
                "best_overlap_fraction": "overlap_fraction",
            }
        )
        pred_channel0_object_rows["stack_id"] = stack_id
        pred_channel0_rows.append(pred_channel0_object_rows)

        gt_lookup = gt_channel0_object_rows.set_index("label")
        pred_lookup = pred_channel0_object_rows.set_index("label")
        for match_row in match_ch0["matches"]:
            gt_label = int(match_row["gt_label"])
            pred_label = int(match_row["pred_label"])
            matched_channel0_rows.append(
                {
                    "stack_id": stack_id,
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "iou": float(match_row["iou"]),
                    "gt_area_px_2d": float(gt_lookup.loc[gt_label, "area_px_2d"]),
                    "pred_area_px_2d": float(pred_lookup.loc[pred_label, "area_px_2d"]),
                    "gt_roundness_2d": float(gt_lookup.loc[gt_label, "roundness_2d"]),
                    "pred_roundness_2d": float(pred_lookup.loc[pred_label, "roundness_2d"]),
                    "gt_eccentricity_2d": float(gt_lookup.loc[gt_label, "eccentricity_2d"]),
                    "pred_eccentricity_2d": float(pred_lookup.loc[pred_label, "eccentricity_2d"]),
                    "gt_overlap_fraction": float(gt_lookup.loc[gt_label, "overlap_fraction"]),
                    "pred_overlap_fraction": float(pred_lookup.loc[pred_label, "overlap_fraction"]),
                }
            )

        stack_rows.append(
            {
                "stack_id": stack_id,
                "gt_channel0_count": int(gt_row["n_channel0_cells_gt"]),
                "pred_channel0_count": int(overview["n_cells"]),
                "gt_channel1_count": int(gt_row["n_channel1_cells_gt"]),
                "pred_channel1_count": int(overview["n_marker_objects"]),
                "gt_colocalized_count": int(gt_row["n_channel0_marker_positive_gt"]),
                "pred_colocalized_count": int(overview["n_marker_positive_cells"]),
                "gt_channel2_coverage_percent": float(gt_row["channel2_coverage_gt_percent"]),
                "pred_channel2_coverage_percent": float(overview["optional_region_occupancy_coverage_2d_percent"]),
                "channel0_precision": float(match_ch0["precision"]),
                "channel0_recall": float(match_ch0["recall"]),
                "channel0_f1": float(match_ch0["f1"]),
                "channel1_precision": float(match_ch1["precision"]),
                "channel1_recall": float(match_ch1["recall"]),
                "channel1_f1": float(match_ch1["f1"]),
            })

    gt_channel0_objects = pd.concat(gt_channel0_rows, ignore_index=True)
    pred_channel0_objects = pd.concat(pred_channel0_rows, ignore_index=True)
    matched_channel0 = pd.DataFrame(matched_channel0_rows)
    detection_long = pd.DataFrame(detection_rows)
    stack_table = pd.DataFrame(stack_rows).sort_values("stack_id").reset_index(drop=True)
    return SyntheticEvaluation(
        stack_table=stack_table,
        detection_long=detection_long,
        gt_channel0_objects=gt_channel0_objects,
        pred_channel0_objects=pred_channel0_objects,
        matched_channel0=matched_channel0)
# %% QUANTITATIVE PLOT HELPERS
def draw_spaghetti_panel(
    stack_table: pd.DataFrame,
    gt_column: str,
    pred_column: str,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    color_map = build_stack_color_map(stack_table["stack_id"].tolist())
    for row in stack_table.itertuples(index=False):
        color = color_map[row.stack_id]
        gt_value = getattr(row, gt_column)
        pred_value = getattr(row, pred_column)
        ax.plot([0, 1], [gt_value, pred_value], color=color, alpha=0.55, linewidth=1.2)
        ax.scatter([0, 1], [gt_value, pred_value], color=color, s=26, alpha=0.9)

    ax.set_xticks([0, 1], ["GT", "CellColoc"])
    apply_axes_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_detection_metrics_panel(detection_long: pd.DataFrame, config: PanelConfig, output_path: Path) -> None:
    fig, ax = create_panel_figure(config)
    metrics = ["precision", "recall", "f1"]
    channels = ["Channel 0", "Channel 1"]
    x_positions = np.arange(len(metrics))
    bar_width = 0.34

    for offset, channel in zip((-bar_width / 2.0, bar_width / 2.0), channels):
        subset = detection_long[detection_long["channel"] == channel]
        means = [float(subset[metric].mean()) for metric in metrics]
        stds = [float(subset[metric].std(ddof=1)) for metric in metrics]
        color = GT_COLOR if channel == "Channel 0" else PRED_COLOR
        ax.bar(
            x_positions + offset,
            means,
            width=bar_width,
            color=color,
            alpha=0.88,
            yerr=stds,
            capsize=3,
            label=channel)

    ax.set_xticks(x_positions, ["Precision", "Recall", "F1"])
    apply_axes_controls(ax, config)
    apply_legend_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_synthetic_error_summary_panel(stack_table: pd.DataFrame, config: PanelConfig, output_path: Path) -> None:
    fig, ax = create_panel_figure(config)
    measures = [
        ("Ch0 count", np.abs(stack_table["pred_channel0_count"] - stack_table["gt_channel0_count"])),
        ("Ch1 count", np.abs(stack_table["pred_channel1_count"] - stack_table["gt_channel1_count"])),
        ("Ch0+Ch1", np.abs(stack_table["pred_colocalized_count"] - stack_table["gt_colocalized_count"])),
        ("Ch2 cov. (pp)", np.abs(stack_table["pred_channel2_coverage_percent"] - stack_table["gt_channel2_coverage_percent"])),
    ]
    x_positions = np.arange(len(measures))
    means = [float(values.mean()) for _, values in measures]
    stds = [float(values.std(ddof=1)) for _, values in measures]

    ax.bar(x_positions, means, color="#5c879d", alpha=0.88, yerr=stds, capsize=3)
    ax.set_xticks(x_positions, [label for label, _ in measures])
    apply_axes_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_gt_vs_pred_distribution_panel(
    gt_data: pd.DataFrame,
    pred_data: pd.DataFrame,
    value_column: str,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    stack_ids = sorted(set(gt_data["stack_id"]).union(set(pred_data["stack_id"])))
    color_map = build_stack_color_map(stack_ids)
    rng = np.random.default_rng(20260709)

    for source_index, (_, data) in enumerate((("GT", gt_data), ("CellColoc", pred_data))):
        for stack_id, subset in data.groupby("stack_id"):
            values = subset[value_column].to_numpy(dtype=float)
            jitter = rng.normal(loc=0.0, scale=0.035, size=len(values))
            ax.scatter(
                np.full(len(values), source_index) + jitter,
                values,
                s=18,
                color=color_map[stack_id],
                alpha=0.72,
                edgecolors="none")

    ax.set_xticks([0, 1], ["GT", "CellColoc"])
    apply_axes_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_matched_scatter_panel(
    matched_data: pd.DataFrame,
    gt_column: str,
    pred_column: str,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    color_map = build_stack_color_map(matched_data["stack_id"].tolist())
    for stack_id, subset in matched_data.groupby("stack_id"):
        ax.scatter(
            subset[gt_column],
            subset[pred_column],
            s=22,
            color=color_map[stack_id],
            alpha=0.72,
            edgecolors="none",
        )

    values_x = matched_data[gt_column].to_numpy(dtype=float)
    values_y = matched_data[pred_column].to_numpy(dtype=float)
    line_min = min(float(values_x.min()), float(values_y.min()))
    line_max = max(float(values_x.max()), float(values_y.max()))
    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="#5d7282", linewidth=1.1)

    mae = float(np.mean(np.abs(values_y - values_x)))
    medae = float(np.median(np.abs(values_y - values_x)))
    corr = safe_pearsonr(values_x, values_y)
    metrics_text = f"n = {len(values_x)}\nMAE = {mae:.2f}\nMedAE = {medae:.2f}\nr = {corr:.3f}"
    ax.text(
        0.04,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d1dbe4", alpha=0.95))

    ax.set_xlabel("GT")
    apply_axes_controls(ax, config)
    finalize_panel(fig, output_path, config)
# %% SYNTHETIC FIGURE GENERATION
def draw_synthetic_benchmark_panels(
    evaluation: SyntheticEvaluation,
    panel_set: dict[str, PanelConfig],
    *,
    paths: SyntheticBenchmarkPaths,
) -> None:
    gt_summary = pd.read_csv(paths.gt_dir / "synthetic_benchmark_ground_truth_summary.csv")
    example_stack_id = choose_synthetic_example_stack(gt_summary)
    example_stack = np.asarray(load_synthetic_stack_image(paths, example_stack_id), dtype=np.float32)
    result_stem = synthetic_result_stem(example_stack_id)

    channel0 = normalize_image_for_display(example_stack[0])
    channel1 = normalize_image_for_display(example_stack[1])
    channel2 = normalize_image_for_display(example_stack[2])
    composite = np.stack(
        [
            np.clip(channel0 + channel2 * 0.7, 0.0, 1.0),
            np.clip(channel1 + channel2 * 0.7, 0.0, 1.0),
            np.clip(channel1, 0.0, 1.0),
        ], axis=-1)

    example_masks = {
        "mask_channel0": read_first_plane(paths.results_dir / f"{result_stem}_cell_masks.tif"),
        "mask_channel1": read_first_plane(paths.results_dir / f"{result_stem}_marker_masks.tif"),
        "mask_region": read_first_plane(paths.results_dir / f"{result_stem}_region_mask.tif"),
        "mask_positive": read_first_plane(paths.results_dir / f"{result_stem}_positive_cell_masks.tif")}

    save_image_panel(channel0, panel_set["raw_channel0"], output_path_for(panel_set["raw_channel0"]))
    save_image_panel(channel1, panel_set["raw_channel1"], output_path_for(panel_set["raw_channel1"]))
    save_image_panel(channel2, panel_set["raw_channel2"], output_path_for(panel_set["raw_channel2"]))
    save_image_panel(composite, panel_set["raw_composite"], output_path_for(panel_set["raw_composite"]))

    for key, mask in example_masks.items():
        save_label_mask_panel(mask, panel_set[key], output_path_for(panel_set[key]))

    draw_spaghetti_panel(
        evaluation.stack_table,
        "gt_channel0_count",
        "pred_channel0_count",
        panel_set["counts_channel0"],
        output_path_for(panel_set["counts_channel0"]))
    draw_spaghetti_panel(
        evaluation.stack_table,
        "gt_channel1_count",
        "pred_channel1_count",
        panel_set["counts_channel1"],
        output_path_for(panel_set["counts_channel1"]))
    draw_spaghetti_panel(
        evaluation.stack_table,
        "gt_colocalized_count",
        "pred_colocalized_count",
        panel_set["counts_positive"],
        output_path_for(panel_set["counts_positive"]))
    draw_spaghetti_panel(
        evaluation.stack_table,
        "gt_channel2_coverage_percent",
        "pred_channel2_coverage_percent",
        panel_set["coverage_channel2"],
        output_path_for(panel_set["coverage_channel2"]))
    draw_detection_metrics_panel(
        evaluation.detection_long,
        panel_set["detection_metrics"],
        output_path_for(panel_set["detection_metrics"]))
    draw_synthetic_error_summary_panel(
        evaluation.stack_table,
        panel_set["error_summary"],
        output_path_for(panel_set["error_summary"]))
    draw_matched_scatter_panel(
        evaluation.matched_channel0,
        "gt_area_px_2d",
        "pred_area_px_2d",
        panel_set["areas_matched"],
        output_path_for(panel_set["areas_matched"]))
    draw_matched_scatter_panel(
        evaluation.matched_channel0,
        "gt_roundness_2d",
        "pred_roundness_2d",
        panel_set["roundness_matched"],
        output_path_for(panel_set["roundness_matched"]))
    draw_matched_scatter_panel(
        evaluation.matched_channel0,
        "gt_eccentricity_2d",
        "pred_eccentricity_2d",
        panel_set["eccentricity_matched"],
        output_path_for(panel_set["eccentricity_matched"]))
    draw_matched_scatter_panel(
        evaluation.matched_channel0,
        "gt_overlap_fraction",
        "pred_overlap_fraction",
        panel_set["overlap_fraction_matched"],
        output_path_for(panel_set["overlap_fraction_matched"]))
    draw_gt_vs_pred_distribution_panel(
        evaluation.gt_channel0_objects,
        evaluation.pred_channel0_objects,
        "overlap_fraction",
        panel_set["overlap_fraction_distribution"],
        output_path_for(panel_set["overlap_fraction_distribution"]))
# %% MICROGLIA HELPERS AND FIGURE GENERATION
def load_microglia_manifest() -> pd.DataFrame:
    if not MICROGLIA_MANIFEST_PATH.exists():
        partial_manifests = sorted(MICROGLIA_FULL_DIR.glob("ca1_manifest*.csv")) + sorted(
            MICROGLIA_FULL_DIR.glob("ctx_manifest*.csv")
        )
        if partial_manifests:
            manifest = pd.concat([pd.read_csv(path) for path in partial_manifests], ignore_index=True)
            manifest = manifest.drop_duplicates(subset=["region", "stack_name"]).sort_values(by=["region", "stack_name"])
            manifest.to_csv(MICROGLIA_MANIFEST_PATH, index=False)
        else:
            raise FileNotFoundError(
                f"Missing {MICROGLIA_MANIFEST_PATH}. "
                "Run additional_scripts/run_microglia_3d_full_batch.py first.")
    manifest = pd.read_csv(MICROGLIA_MANIFEST_PATH)
    if manifest.empty:
        raise ValueError("The microglia batch manifest is empty.")
    enriched_manifest = enrich_microglia_manifest(manifest)
    if set(enriched_manifest.columns) != set(manifest.columns):
        enriched_manifest.to_csv(MICROGLIA_MANIFEST_PATH, index=False)
    return enriched_manifest

def resolve_microglia_source_path(source_path: Path) -> Path:
    if source_path.exists():
        return source_path
    source_path_text = str(source_path)
    for legacy_id, current_id in MICROGLIA_MOUSE_ID_ALIASES.items():
        candidate = Path(source_path_text.replace(f"ID{legacy_id}", f"ID{current_id}"))
        if candidate.exists():
            return candidate
    return source_path

def compute_stack_volume_mm3(source_path: Path) -> float:
    loaded_images = load_analysis_images(
        source_path=source_path,
        channel_config=MICROGLIA_CHANNEL_CONFIG,
        voxel_scale_zyx=None,
        image_loading_mode="memap",
    )
    z_size_px, y_size_px, x_size_px = loaded_images.cell_image.shape
    z_size_um, y_size_um, x_size_um = loaded_images.voxel_scale_zyx
    volume_um3 = float(z_size_px * y_size_px * x_size_px) * float(z_size_um * y_size_um * x_size_um)
    return volume_um3 / 1e9

def load_mean_cell_eccentricity(results_dir: Path, stack_name: str) -> float:
    summary_path = results_dir / f"{Path(stack_name).stem}_cell_summary.csv"
    if not summary_path.exists():
        return float("nan")
    summary_table = pd.read_csv(summary_path)
    if "cell_eccentricity_2d" not in summary_table.columns or summary_table.empty:
        return float("nan")
    return float(pd.to_numeric(summary_table["cell_eccentricity_2d"], errors="coerce").dropna().mean())

def load_mean_cell_area_um2(results_dir: Path, stack_name: str) -> float:
    summary_path = results_dir / f"{Path(stack_name).stem}_cell_summary.csv"
    if not summary_path.exists():
        return float("nan")
    summary_table = pd.read_csv(summary_path)
    if "cell_area_um2_2d" not in summary_table.columns or summary_table.empty:
        return float("nan")
    return float(pd.to_numeric(summary_table["cell_area_um2_2d"], errors="coerce").dropna().mean())

def load_mean_cell_brightness(results_dir: Path, stack_name: str, source_path: Path) -> float:
    stack_stem = Path(stack_name).stem
    summary_path = results_dir / f"{stack_stem}_cell_summary.csv"
    if summary_path.exists():
        summary_table = pd.read_csv(summary_path)
        if "cell_mean_intensity" in summary_table.columns and not summary_table.empty:
            values = pd.to_numeric(summary_table["cell_mean_intensity"], errors="coerce").dropna()
            if not values.empty:
                return float(values.mean())

    mask_path = results_dir / f"{stack_stem}_cell_masks.tif"
    if not mask_path.exists() or not source_path.exists():
        return float("nan")

    masks = np.asarray(tifffile.imread(mask_path))
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    if masks.ndim != 3:
        return float("nan")

    loaded_images = load_analysis_images(
        source_path=source_path,
        channel_config=MICROGLIA_CHANNEL_CONFIG,
        voxel_scale_zyx=None,
        crop_for_testing=None,
        image_loading_mode="memap")
    cell_image = np.asarray(loaded_images.cell_image, dtype=np.float32)
    if cell_image.shape != masks.shape:
        if masks.shape[0] != 1:
            return float("nan")
        config_path = results_dir / f"{stack_stem}_analysis_config.json"
        projection_method = "max"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                analysis_config = json.load(handle)
            projection_method = str(analysis_config.get("cell_model_config", {}).get("z_projection") or "max")
        cell_image = project_scalar_volume(
            cell_image,
            projection_method=projection_method,
            z_slice_index=None)[np.newaxis, ...]

    if cell_image.shape != masks.shape:
        return float("nan")

    labels = np.unique(masks)
    labels = labels[labels != 0]
    if labels.size == 0:
        return float("nan")
    object_means = [float(cell_image[masks == label].mean()) for label in labels]
    return float(np.mean(object_means))

def enrich_microglia_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    if "source_path" in manifest.columns:
        manifest["source_path"] = manifest["source_path"].map(lambda value: str(resolve_microglia_source_path(Path(str(value)))))
    required_columns = {
        "stack_volume_mm3",
        "n_cells_per_mm3",
        "n_iba1_positive_cells_per_mm3",
        "mean_cell_area_um2_2d",
        "mean_cell_eccentricity_2d",
        "mean_cell_brightness"}
    if required_columns.issubset(manifest.columns):
        if manifest[list(required_columns)].notna().all().all():
            return manifest

    stack_volumes_mm3: list[float] = []
    mean_areas_um2: list[float] = []
    mean_eccentricities: list[float] = []
    mean_cell_brightnesses: list[float] = []
    for _, row in manifest.iterrows():
        source_path = resolve_microglia_source_path(Path(str(row["source_path"])))
        results_dir = Path(str(row["results_dir"]))
        stack_name = str(row["stack_name"])
        stack_volumes_mm3.append(compute_stack_volume_mm3(source_path))
        mean_areas_um2.append(load_mean_cell_area_um2(results_dir, stack_name))
        mean_eccentricities.append(load_mean_cell_eccentricity(results_dir, stack_name))
        mean_cell_brightnesses.append(load_mean_cell_brightness(results_dir, stack_name, source_path))

    manifest["stack_volume_mm3"] = stack_volumes_mm3
    manifest["n_cells_per_mm3"] = manifest["n_cells"].astype(float) / manifest["stack_volume_mm3"].astype(float)
    manifest["n_iba1_positive_cells_per_mm3"] = (
        manifest["n_iba1_positive_cells"].astype(float) / manifest["stack_volume_mm3"].astype(float)
    )
    manifest["mean_cell_area_um2_2d"] = mean_areas_um2
    manifest["mean_cell_eccentricity_2d"] = mean_eccentricities
    manifest["mean_cell_brightness"] = mean_cell_brightnesses
    return manifest

def extract_microglia_mouse_id(stack_name: str) -> str:
    match = re.search(r"ID(\d+)", str(stack_name))
    if match is None:
        raise ValueError(f"Could not extract a mouse ID from stack name {stack_name!r}.")
    mouse_id = match.group(1)
    return MICROGLIA_MOUSE_ID_ALIASES.get(mouse_id, mouse_id)

def aggregate_microglia_manifest_by_mouse_region(manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    manifest["region"] = manifest["region"].astype(str).str.upper()
    manifest = manifest.loc[manifest["region"].isin(["CTX", "CA1"])].copy()
    manifest["mouse_id"] = manifest["stack_name"].map(extract_microglia_mouse_id)
    manifest["treatment"] = manifest["mouse_id"].map(MICROGLIA_TREATMENT_BY_MOUSE_ID).fillna("Unknown")

    numeric_columns = [
        column
        for column in [
            "n_cells",
            "n_iba1_positive_cells",
            "iba1_positive_fraction",
            "mean_cell_area_px_2d",
            "mean_cell_roundness_2d",
            "stack_volume_mm3",
            "n_cells_per_mm3",
            "n_iba1_positive_cells_per_mm3",
            "mean_cell_area_um2_2d",
            "mean_cell_eccentricity_2d",
            "mean_cell_brightness"]
        if column in manifest.columns]
    aggregation_map: dict[str, str | callable] = {column: "mean" for column in numeric_columns}
    aggregation_map.update(
        {
            "stack_name": lambda values: "; ".join(sorted(str(value) for value in values)),
            "source_path": "first",
            "results_dir": "first",
            "treatment": "first",
        })
    grouped = (
        manifest.groupby(["mouse_id", "region"], as_index=False)
        .agg(aggregation_map)
        .sort_values(by=["mouse_id", "region"])
        .reset_index(drop=True))
    scan_counts = (
        manifest.groupby(["mouse_id", "region"])
        .size()
        .rename("n_region_scans")
        .reset_index())
    grouped = grouped.merge(scan_counts, on=["mouse_id", "region"], how="left")
    return grouped

def aggregate_paired_microglia_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    grouped = aggregate_microglia_manifest_by_mouse_region(manifest)

    regions_by_mouse = grouped.groupby("mouse_id")["region"].agg(lambda values: set(values))
    paired_mouse_ids = sorted(mouse_id for mouse_id, regions in regions_by_mouse.items() if regions == {"CTX", "CA1"})
    paired_manifest = grouped[grouped["mouse_id"].isin(paired_mouse_ids)].copy()
    if paired_manifest.empty:
        raise ValueError("No paired CTX/CA1 microglia mice were found in the manifest.")
    return paired_manifest

def aggregate_microglia_manifest_by_mouse_treatment(manifest: pd.DataFrame) -> pd.DataFrame:
    grouped = aggregate_microglia_manifest_by_mouse_region(manifest)
    numeric_columns = [
        column
        for column in [
            "n_cells_per_mm3",
            "n_iba1_positive_cells_per_mm3",
            "iba1_positive_fraction",
            "mean_cell_area_um2_2d",
            "mean_cell_roundness_2d",
            "mean_cell_eccentricity_2d",
            "mean_cell_brightness",
        ]
        if column in grouped.columns
    ]
    aggregation_map: dict[str, str | callable] = {column: "mean" for column in numeric_columns}
    aggregation_map.update(
        {
            "region": lambda values: "; ".join(sorted(str(value) for value in values)),
            "n_region_scans": "sum",
        })
    mouse_level = (
        grouped.groupby(["mouse_id", "treatment"], as_index=False)
        .agg(aggregation_map)
        .sort_values(by=["treatment", "mouse_id"])
        .reset_index(drop=True))
    mouse_level["n_regions"] = mouse_level["region"].str.count(";") + 1
    return mouse_level

def choose_representative_microglia_row(manifest: pd.DataFrame, region: str) -> pd.Series:
    subset = manifest[manifest["region"] == region].copy()
    target = float(subset["iba1_positive_fraction"].median())
    distances = (subset["iba1_positive_fraction"] - target).abs()
    return subset.loc[distances.idxmin()]

def format_p_value(stats_row: dict[str, float | str]) -> str:
    value = float(stats_row["p_value_display"])
    value_label = str(stats_row["p_value_label"])
    if value < 1e-3:
        return f"{value_label} < 0.001"
    return f"{value_label} = {value:.3f}"

def compute_microglia_statistics(
    manifest: pd.DataFrame,
    *,
    apply_holm_correction: bool,
) -> dict[str, dict[str, float]]:
    metric_columns = [
        "n_cells_per_mm3",
        "n_iba1_positive_cells_per_mm3",
        "iba1_positive_fraction",
        "mean_cell_area_um2_2d",
        "mean_cell_roundness_2d",
        "mean_cell_eccentricity_2d",
        "mean_cell_brightness"]
    raw_p_values: list[float] = []
    rows: list[dict[str, float | str]] = []
    for metric_column in metric_columns:
        paired_values = (
            manifest.pivot(index="mouse_id", columns="region", values=metric_column)
            .dropna(subset=["CTX", "CA1"])
            .sort_index()
        )
        ctx_values = paired_values["CTX"].to_numpy(dtype=float)
        ca1_values = paired_values["CA1"].to_numpy(dtype=float)
        differences = ca1_values - ctx_values

        normality_p_value = float("nan")
        use_paired_t = False
        if len(differences) >= 3 and not np.allclose(differences, differences[0]):
            normality_p_value = float(shapiro(differences).pvalue)
            use_paired_t = normality_p_value > 0.05

        if use_paired_t:
            statistic, p_value = ttest_rel(ca1_values, ctx_values, nan_policy="omit")
            test_name = "paired t-test"
            test_label = "paired t"
        else:
            if np.allclose(differences, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(ca1_values, ctx_values, alternative="two-sided")
            test_name = "Wilcoxon signed-rank"
            test_label = "Wilcoxon"

        rows.append(
            {
                "metric": metric_column,
                "statistic": float(statistic),
                "p_value": float(p_value),
                "n_pairs": float(len(paired_values)),
                "normality_p_value": normality_p_value,
                "test_name": test_name,
                "test_label": test_label})
        raw_p_values.append(float(p_value))

    adjusted = holm_adjust(raw_p_values)
    stats_map: dict[str, dict[str, float]] = {}
    for row, adjusted_p in zip(rows, adjusted):
        display_p_value = float(adjusted_p) if apply_holm_correction else float(row["p_value"])
        display_label = "Holm p" if apply_holm_correction else "p"
        stats_map[str(row["metric"])] = {
            "statistic": float(row["statistic"]),
            "p_value": float(row["p_value"]),
            "p_value_holm": float(adjusted_p),
            "p_value_display": display_p_value,
            "p_value_label": display_label,
            "n_pairs": float(row["n_pairs"]),
            "normality_p_value": float(row["normality_p_value"]),
            "test_name": str(row["test_name"]),
            "test_label": str(row["test_label"])}
    return stats_map

def add_p_value_bracket(ax: plt.Axes, x0: float, x1: float, y: float, text: str, height: float) -> None:
    ax.plot([x0, x0, x1, x1], [y, y + height, y + height, y], color="#4c6070", linewidth=1.0)
    ax.text((x0 + x1) / 2.0, y + height * 1.08, text, ha="center", va="bottom", fontsize=8.3)

def draw_group_comparison_panel(
    manifest: pd.DataFrame,
    metric_column: str,
    stats_map: dict[str, dict[str, float]],
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    paired_values = (
        manifest.pivot(index="mouse_id", columns="region", values=metric_column)
        .dropna(subset=["CTX", "CA1"])
        .sort_index())
    regions = ["CTX", "CA1"]
    colors = [CTX_COLOR, CA1_COLOR]
    data = [paired_values[region].to_numpy(dtype=float) for region in regions]

    box = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)
    for median in box["medians"]:
        median.set_color("#1f2e3c")
        median.set_linewidth(1.4)

    rng = np.random.default_rng(20260709)
    pair_jitter = rng.normal(loc=0.0, scale=0.03, size=len(paired_values))
    for index, (values, color) in enumerate(zip(data, colors), start=1):
        ax.scatter(
            np.full(len(values), index, dtype=float) + pair_jitter,
            values,
            s=28,
            color=color,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    for jitter_value, (_, row) in zip(pair_jitter, paired_values.iterrows()):
        ax.plot(
            [1.0 + jitter_value, 2.0 + jitter_value],
            [float(row["CTX"]), float(row["CA1"])],
            color="#a8b3bc",
            linewidth=0.9,
            alpha=0.9,
            zorder=2)

    n_pairs = len(paired_values)
    ax.set_xticks([1, 2], [f"CTX\n(n={n_pairs})", f"CA1\n(n={n_pairs})"])

    all_values = np.concatenate(data)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    lower = config.ylim[0] if config.ylim is not None and config.ylim[0] is not None else max(0.0, y_min * 0.92)
    upper = config.ylim[1] if config.ylim is not None and config.ylim[1] is not None else (y_max * 1.22 if y_max > 0 else 1.0)
    ax.set_ylim(lower, upper)

    p_text = format_p_value(stats_map[metric_column])
    bracket_y = upper - (upper - lower) * 0.12
    bracket_height = (upper - lower) * 0.035
    add_p_value_bracket(ax, 1, 2, bracket_y, p_text, bracket_height)

    apply_axes_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_treatment_region_panel(
    manifest: pd.DataFrame,
    metric_column: str,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)

    group_specs = [
        ("CTX", "Vehicle", 1.0, CTX_VEHICLE_COLOR),
        ("CTX", "Tamoxifen", 2.0, CTX_TAMOX_COLOR),
        ("CA1", "Vehicle", 4.0, CA1_VEHICLE_COLOR),
        ("CA1", "Tamoxifen", 5.0, CA1_TAMOX_COLOR),
    ]
    label_map = {
        ("CTX", "Vehicle"): f"CTX\n{MICROGLIA_TREATMENT_XTICK_LABELS['Vehicle']}",
        ("CTX", "Tamoxifen"): f"CTX\n{MICROGLIA_TREATMENT_XTICK_LABELS['Tamoxifen']}",
        ("CA1", "Vehicle"): f"CA1\n{MICROGLIA_TREATMENT_XTICK_LABELS['Vehicle']}",
        ("CA1", "Tamoxifen"): f"CA1\n{MICROGLIA_TREATMENT_XTICK_LABELS['Tamoxifen']}",
    }
    position_map = {(region, treatment): position for region, treatment, position, _ in group_specs}

    grouped_data: list[np.ndarray] = []
    for region, treatment, _, _ in group_specs:
        values = (
            manifest.loc[
                (manifest["region"] == region) & (manifest["treatment"] == treatment),
                metric_column,
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        grouped_data.append(values)

    positions = [position for _, _, position, _ in group_specs]
    valid_data = [values if len(values) else np.array([np.nan]) for values in grouped_data]
    box = ax.boxplot(valid_data, positions=positions, patch_artist=True, widths=0.55, showfliers=False)
    for patch, (_, _, _, color) in zip(box["boxes"], group_specs):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)
    for median in box["medians"]:
        median.set_color("#1f2e3c")
        median.set_linewidth(1.4)

    rng = np.random.default_rng(20260722)
    for values, (region, treatment, position, color) in zip(grouped_data, group_specs):
        if not len(values):
            continue
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
        ax.scatter(
            np.full(len(values), position, dtype=float) + jitter,
            values,
            s=30,
            color=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.3,
            zorder=3)

    paired_subset = manifest.copy()
    for treatment, line_color in [("Vehicle", "#b3bcc4"), ("Tamoxifen", "#8f99a3")]:
        paired_values = (
            paired_subset.loc[paired_subset["treatment"] == treatment]
            .pivot(index="mouse_id", columns="region", values=metric_column)
            .dropna(subset=["CTX", "CA1"])
            .sort_index()
        )
        if paired_values.empty:
            continue
        line_jitter = rng.normal(loc=0.0, scale=0.03, size=len(paired_values))
        for jitter_value, (_, row) in zip(line_jitter, paired_values.iterrows()):
            ax.plot(
                [
                    position_map[("CTX", treatment)] + jitter_value,
                    position_map[("CA1", treatment)] + jitter_value,
                ],
                [float(row["CTX"]), float(row["CA1"])],
                color=line_color,
                linewidth=0.9,
                alpha=0.95,
                zorder=2)

    all_values = np.concatenate([values for values in grouped_data if len(values)]) if any(len(values) for values in grouped_data) else np.array([0.0])
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    lower = config.ylim[0] if config.ylim is not None and config.ylim[0] is not None else max(0.0, y_min * 0.92)
    upper = config.ylim[1] if config.ylim is not None and config.ylim[1] is not None else (y_max * 1.22 if y_max > 0 else 1.0)
    ax.set_ylim(lower, upper)

    tick_labels = []
    for region, treatment, _, _ in group_specs:
        n_group = int(((manifest["region"] == region) & (manifest["treatment"] == treatment)).sum())
        tick_labels.append(f"{label_map[(region, treatment)]}\n(n={n_group})")
    ax.set_xticks(positions, tick_labels)
    ax.axvline(3.0, color="#d0d7dd", linewidth=0.8, linestyle="--", zorder=1)

    apply_axes_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_global_treatment_panel(
    manifest: pd.DataFrame,
    metric_column: str,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)

    group_specs = [
        ("Vehicle", 1.0, "#86c7d4"),
        ("Tamoxifen", 2.0, "#e95f4e")]
    data = [
        manifest.loc[manifest["treatment"] == treatment, metric_column].dropna().to_numpy(dtype=float)
        for treatment, _, _ in group_specs]

    box = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False)
    for patch, (_, _, color) in zip(box["boxes"], group_specs):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)
    for median in box["medians"]:
        median.set_color("#1f2e3c")
        median.set_linewidth(1.4)

    rng = np.random.default_rng(20260722)
    for values, (_, position, color) in zip(data, group_specs):
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
        ax.scatter(
            np.full(len(values), position, dtype=float) + jitter,
            values,
            s=34,
            color=color,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.3,
            zorder=3)

    all_values = np.concatenate([values for values in data if len(values)]) if any(len(values) for values in data) else np.array([0.0])
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    lower = config.ylim[0] if config.ylim is not None and config.ylim[0] is not None else max(0.0, y_min * 0.92)
    upper = config.ylim[1] if config.ylim is not None and config.ylim[1] is not None else (y_max * 1.25 if y_max > 0 else 1.0)
    ax.set_ylim(lower, upper)

    labels = []
    for treatment, _, _ in group_specs:
        n_group = int((manifest["treatment"] == treatment).sum())
        treatment_label = MICROGLIA_TREATMENT_XTICK_LABELS.get(treatment, treatment)
        labels.append(f"{treatment_label}\n(n={n_group})")
    ax.set_xticks([1, 2], labels)

    if all(len(values) > 0 for values in data):
        _, p_value = mannwhitneyu(data[0], data[1], alternative="two-sided", method="auto")
        p_text = "p < 0.001" if p_value < 1e-3 else f"p = {p_value:.3f}"
        bracket_y = upper - (upper - lower) * 0.12
        bracket_height = (upper - lower) * 0.035
        add_p_value_bracket(ax, 1, 2, bracket_y, p_text, bracket_height)

    apply_axes_controls(ax, config)
    finalize_panel(fig, output_path, config)

def get_paired_region_metric_values(manifest: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    return (
        manifest.pivot(index="mouse_id", columns="region", values=metric_column)
        .dropna(subset=["CTX", "CA1"])
        .sort_index())

def draw_morphology_effect_raw_panel(
    manifest: pd.DataFrame,
    config: PanelConfig,
    output_path: Path,
) -> None:
    width_cm, height_cm = config.figsize_cm
    fig, axes = plt.subplots(
        nrows=len(MICROGLIA_MORPHOLOGY_EFFECT_METRICS),
        ncols=1,
        figsize=(cm_to_inch(width_cm), cm_to_inch(height_cm)),
        dpi=PANEL_DPI)
    rng = np.random.default_rng(20260724)
    for ax, (metric_column, metric_label, unit_label) in zip(axes, MICROGLIA_MORPHOLOGY_EFFECT_METRICS):
        paired_values = get_paired_region_metric_values(manifest, metric_column)
        differences = (paired_values["CA1"] - paired_values["CTX"]).to_numpy(dtype=float)
        y_values = rng.normal(loc=0.0, scale=0.035, size=len(differences))
        ax.axvline(0.0, color="#7c8a95", linewidth=0.9, linestyle="--", zorder=1)
        ax.scatter(
            differences,
            y_values,
            s=30,
            color=CA1_COLOR,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.3,
            zorder=3)
        mean_difference = float(np.mean(differences))
        sem_difference = float(np.std(differences, ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else 0.0
        ax.errorbar(
            mean_difference,
            0.22,
            xerr=sem_difference,
            fmt="o",
            color="#1f2e3c",
            ecolor="#1f2e3c",
            elinewidth=1.2,
            capsize=3,
            markersize=4,
            zorder=4)
        unit_suffix = f" ({unit_label})" if unit_label else ""
        ax.set_ylabel(metric_label, rotation=0, ha="right", va="center", labelpad=40)
        ax.set_xlabel(f"CA1 - CTX{unit_suffix}")
        ax.set_yticks([])
        ax.set_ylim(-0.22, 0.42)
        for spine_name, visible in config.spines.items():
            ax.spines[spine_name].set_visible(visible)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
        ax.tick_params(axis="x", length=3)
    axes[0].set_title(config.title, fontsize=10, pad=config.title_pad)
    finalize_panel(fig, output_path, config)

def draw_morphology_effect_standardized_panel(
    manifest: pd.DataFrame,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    rng = np.random.default_rng(20260724)
    y_positions = np.arange(len(MICROGLIA_MORPHOLOGY_EFFECT_METRICS))[::-1]
    y_labels: list[str] = []
    for y_position, (metric_column, metric_label, _) in zip(y_positions, MICROGLIA_MORPHOLOGY_EFFECT_METRICS):
        paired_values = get_paired_region_metric_values(manifest, metric_column)
        differences = (paired_values["CA1"] - paired_values["CTX"]).to_numpy(dtype=float)
        sd_difference = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
        standardized = differences / sd_difference if sd_difference > 0 else np.zeros_like(differences)
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(standardized))
        ax.scatter(
            standardized,
            np.full(len(standardized), y_position, dtype=float) + jitter,
            s=30,
            color=CA1_COLOR,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.3,
            zorder=3)
        mean_standardized = float(np.mean(standardized))
        sem_standardized = float(np.std(standardized, ddof=1) / np.sqrt(len(standardized))) if len(standardized) > 1 else 0.0
        ax.errorbar(
            mean_standardized,
            y_position + 0.22,
            xerr=sem_standardized,
            fmt="o",
            color="#1f2e3c",
            ecolor="#1f2e3c",
            elinewidth=1.2,
            capsize=3,
            markersize=4,
            zorder=4)
        y_labels.append(metric_label)
    ax.axvline(0.0, color="#7c8a95", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_yticks(y_positions, y_labels)
    apply_axes_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_microglia_qc_density_positive_fraction_panel(
    manifest: pd.DataFrame,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    paired_values = (
        manifest.pivot(index="mouse_id", columns="region", values=["n_cells_per_mm3", "iba1_positive_fraction"])
        .dropna()
        .sort_index()
    )
    for _, row in paired_values.iterrows():
        ax.plot(
            [float(row[("n_cells_per_mm3", "CTX")]), float(row[("n_cells_per_mm3", "CA1")])],
            [float(row[("iba1_positive_fraction", "CTX")]), float(row[("iba1_positive_fraction", "CA1")])],
            color="#a8b3bc",
            linewidth=0.9,
            alpha=0.85,
            zorder=1)
    for region, color in [("CTX", CTX_COLOR), ("CA1", CA1_COLOR)]:
        subset = manifest.loc[manifest["region"] == region]
        ax.scatter(
            subset["n_cells_per_mm3"].to_numpy(dtype=float),
            subset["iba1_positive_fraction"].to_numpy(dtype=float),
            s=34,
            color=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.35,
            label=region,
            zorder=3)
    values = manifest[["n_cells_per_mm3", "iba1_positive_fraction"]].dropna()
    if len(values) >= 3:
        r_value, p_value = pearsonr(values["n_cells_per_mm3"], values["iba1_positive_fraction"])
        p_text = "p < 0.001" if p_value < 1e-3 else f"p = {p_value:.3f}"
        ax.text(
            0.04,
            0.96,
            f"r = {r_value:.2f}\n{p_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d4dde5", "alpha": 0.85})
    apply_axes_controls(ax, config)
    apply_legend_controls(ax, config)
    finalize_panel(fig, output_path, config)

def load_microglia_cell_object_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, manifest_row in manifest.iterrows():
        stack_name = str(manifest_row["stack_name"])
        stack_stem = Path(stack_name).stem
        results_dir = Path(str(manifest_row["results_dir"]))
        summary_path = results_dir / f"{stack_stem}_cell_summary.csv"
        if not summary_path.exists():
            continue
        table = pd.read_csv(summary_path)
        if table.empty or "cell_area_px_2d" not in table.columns:
            continue
        config_path = results_dir / f"{stack_stem}_analysis_config.json"
        min_cell_voxels = float("nan")
        overlap_fraction_threshold = float("nan")
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                analysis_config = json.load(handle)
            colocalization_config = analysis_config.get("colocalization_config", {})
            min_cell_voxels = float(colocalization_config.get("min_cell_voxels", np.nan))
            overlap_fraction_threshold = float(colocalization_config.get("overlap_fraction_threshold", np.nan))
        table = table.copy()
        table["stack_name"] = stack_name
        table["region"] = str(manifest_row["region"]).upper()
        table["source_path"] = str(manifest_row["source_path"])
        table["results_dir"] = str(results_dir)
        table["mouse_id"] = extract_microglia_mouse_id(stack_name)
        table["treatment"] = table["mouse_id"].map(MICROGLIA_TREATMENT_BY_MOUSE_ID).fillna("Unknown")
        table["min_cell_voxels"] = min_cell_voxels
        table["overlap_fraction_threshold"] = overlap_fraction_threshold
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    object_table = pd.concat(rows, ignore_index=True)
    numeric_columns = [
        "cell_area_px_2d",
        "cell_area_um2_2d",
        "cell_roundness_2d",
        "cell_eccentricity_2d",
        "min_cell_voxels",
        "best_overlap_fraction",
        "overlap_fraction_threshold"]
    for column in numeric_columns:
        if column in object_table.columns:
            object_table[column] = pd.to_numeric(object_table[column], errors="coerce")
    return object_table

def microglia_overlap_fraction_threshold(object_table: pd.DataFrame) -> float | None:
    if object_table.empty or "overlap_fraction_threshold" not in object_table.columns:
        return None
    values = object_table["overlap_fraction_threshold"].dropna().unique()
    if len(values) == 0:
        return None
    return float(np.min(values))

def microglia_min_cell_voxel_threshold(object_table: pd.DataFrame) -> float | None:
    if object_table.empty or "min_cell_voxels" not in object_table.columns:
        return None
    values = object_table["min_cell_voxels"].dropna().unique()
    if len(values) == 0:
        return None
    return float(np.min(values))

def add_min_cell_threshold_line(ax: plt.Axes, threshold: float | None, *, orientation: str) -> None:
    if threshold is None:
        return
    if orientation == "horizontal":
        ax.axhline(threshold, color="#4c6070", linestyle="--", linewidth=1.0, zorder=1)
        ax.text(
            0.02,
            threshold,
            f" min={threshold:g} px",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7.8,
            color="#4c6070")
        return
    ax.axvline(threshold, color="#4c6070", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(
        threshold,
        0.96,
        f"min={threshold:g} px",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7.8,
        color="#4c6070",
        rotation=90)

def draw_cell_size_distribution_violin_panel(
    object_table: pd.DataFrame,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    regions = ["CTX", "CA1"]
    colors = [CTX_COLOR, CA1_COLOR]
    data = [
        object_table.loc[object_table["region"] == region, "cell_area_px_2d"].dropna().to_numpy(dtype=float)
        for region in regions
    ]
    violin = ax.violinplot(data, positions=[1, 2], widths=0.65, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)
    if "cmedians" in violin:
        violin["cmedians"].set_color("#1f2e3c")
        violin["cmedians"].set_linewidth(1.2)
    rng = np.random.default_rng(20260724)
    for index, (values, color) in enumerate(zip(data, colors), start=1):
        if not len(values):
            continue
        sampled = values if len(values) <= 450 else rng.choice(values, size=450, replace=False)
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(sampled))
        ax.scatter(
            np.full(len(sampled), index, dtype=float) + jitter,
            sampled,
            s=7,
            color=color,
            alpha=0.18,
            linewidths=0,
            zorder=3)
    ax.set_xticks([1, 2], [f"CTX\n(n={len(data[0])})", f"CA1\n(n={len(data[1])})"])
    add_min_cell_threshold_line(ax, microglia_min_cell_voxel_threshold(object_table), orientation="horizontal")
    apply_axes_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_cell_size_distribution_strip_panel(
    object_table: pd.DataFrame,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    rng = np.random.default_rng(20260724)
    for index, (region, color) in enumerate([("CTX", CTX_COLOR), ("CA1", CA1_COLOR)], start=1):
        values = object_table.loc[object_table["region"] == region, "cell_area_px_2d"].dropna().to_numpy(dtype=float)
        jitter = rng.normal(loc=0.0, scale=0.07, size=len(values))
        ax.scatter(
            np.full(len(values), index, dtype=float) + jitter,
            values,
            s=8,
            color=color,
            alpha=0.20,
            linewidths=0,
            zorder=3)
        if len(values):
            median_value = float(np.median(values))
            ax.plot([index - 0.22, index + 0.22], [median_value, median_value], color="#1f2e3c", linewidth=1.4, zorder=4)
    n_ctx = int((object_table["region"] == "CTX").sum())
    n_ca1 = int((object_table["region"] == "CA1").sum())
    ax.set_xticks([1, 2], [f"CTX\n(n={n_ctx})", f"CA1\n(n={n_ca1})"])
    add_min_cell_threshold_line(ax, microglia_min_cell_voxel_threshold(object_table), orientation="horizontal")
    apply_axes_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_cell_size_distribution_histogram_panel(
    object_table: pd.DataFrame,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    values = object_table["cell_area_px_2d"].dropna().to_numpy(dtype=float)
    upper = float(np.percentile(values, 99.0)) if len(values) else 1.0
    bins = np.linspace(0, max(upper, 1.0), 32)
    for region, color in [("CTX", CTX_COLOR), ("CA1", CA1_COLOR)]:
        region_values = object_table.loc[object_table["region"] == region, "cell_area_px_2d"].dropna().to_numpy(dtype=float)
        ax.hist(
            region_values,
            bins=bins,
            histtype="stepfilled",
            alpha=0.25,
            color=color,
            edgecolor=color,
            linewidth=1.1,
            label=f"{region} (n={len(region_values)})")
    add_min_cell_threshold_line(ax, microglia_min_cell_voxel_threshold(object_table), orientation="vertical")
    apply_axes_controls(ax, config)
    apply_legend_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_area_roundness_scatter_panel(
    object_table: pd.DataFrame,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    for region, color in [("CTX", CTX_COLOR), ("CA1", CA1_COLOR)]:
        subset = object_table.loc[object_table["region"] == region].dropna(subset=["cell_area_px_2d", "cell_roundness_2d"])
        ax.scatter(
            subset["cell_area_px_2d"].to_numpy(dtype=float),
            subset["cell_roundness_2d"].to_numpy(dtype=float),
            s=8,
            color=color,
            alpha=0.18,
            linewidths=0,
            label=f"{region} (n={len(subset)})")
    add_min_cell_threshold_line(ax, microglia_min_cell_voxel_threshold(object_table), orientation="vertical")
    apply_axes_controls(ax, config)
    apply_legend_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_count_median_area_scatter_panel(
    manifest: pd.DataFrame,
    object_table: pd.DataFrame,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    manifest = manifest.copy()
    if "mouse_id" not in manifest.columns:
        manifest["mouse_id"] = manifest["stack_name"].map(extract_microglia_mouse_id)
    median_area = (
        object_table.groupby("stack_name")["cell_area_px_2d"]
        .median()
        .rename("median_cell_area_px_2d")
        .reset_index())
    stack_table = manifest.merge(median_area, on="stack_name", how="left")
    paired_values = (
        stack_table.groupby(["mouse_id", "region"], as_index=False)
        .agg({"n_cells_per_mm3": "mean", "median_cell_area_px_2d": "mean"})
        .pivot(index="mouse_id", columns="region", values=["n_cells_per_mm3", "median_cell_area_px_2d"])
        .dropna()
        .sort_index())
    for _, row in paired_values.iterrows():
        ax.plot(
            [float(row[("n_cells_per_mm3", "CTX")]), float(row[("n_cells_per_mm3", "CA1")])],
            [float(row[("median_cell_area_px_2d", "CTX")]), float(row[("median_cell_area_px_2d", "CA1")])],
            color="#a8b3bc",
            linewidth=0.9,
            alpha=0.85,
            zorder=1)
    for region, color in [("CTX", CTX_COLOR), ("CA1", CA1_COLOR)]:
        subset = stack_table.loc[stack_table["region"] == region].dropna(
            subset=["n_cells_per_mm3", "median_cell_area_px_2d"])
        ax.scatter(
            subset["n_cells_per_mm3"].to_numpy(dtype=float),
            subset["median_cell_area_px_2d"].to_numpy(dtype=float),
            s=34,
            color=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.35,
            label=region,
            zorder=3)
    apply_axes_controls(ax, config)
    apply_legend_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_iba1_overlap_fraction_histogram_panel(
    object_table: pd.DataFrame,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    bins = np.linspace(0.0, 1.0, 31)
    for region, color in [("CTX", CTX_COLOR), ("CA1", CA1_COLOR)]:
        values = object_table.loc[object_table["region"] == region, "best_overlap_fraction"].dropna().to_numpy(dtype=float)
        ax.hist(
            values,
            bins=bins,
            histtype="stepfilled",
            alpha=0.28,
            color=color,
            edgecolor=color,
            linewidth=1.1,
            label=f"{region} (n={len(values)})")
    threshold = microglia_overlap_fraction_threshold(object_table)
    if threshold is not None:
        ax.axvline(threshold, color="#4c6070", linestyle="--", linewidth=1.0, zorder=3)
        ax.text(
            threshold,
            0.96,
            f"threshold={threshold:g}",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=7.8,
            color="#4c6070",
            rotation=90)
    apply_axes_controls(ax, config)
    apply_legend_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_iba1_overlap_fraction_distribution_panel(
    object_table: pd.DataFrame,
    config: PanelConfig,
    output_path: Path,
) -> None:
    fig, ax = create_panel_figure(config)
    regions = ["CTX", "CA1"]
    colors = [CTX_COLOR, CA1_COLOR]
    data = [
        object_table.loc[object_table["region"] == region, "best_overlap_fraction"].dropna().to_numpy(dtype=float)
        for region in regions]
    violin = ax.violinplot(data, positions=[1, 2], widths=0.65, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)
    if "cmedians" in violin:
        violin["cmedians"].set_color("#1f2e3c")
        violin["cmedians"].set_linewidth(1.2)
    rng = np.random.default_rng(20260724)
    for index, (values, color) in enumerate(zip(data, colors), start=1):
        sampled = values if len(values) <= 500 else rng.choice(values, size=500, replace=False)
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(sampled))
        ax.scatter(
            np.full(len(sampled), index, dtype=float) + jitter,
            sampled,
            s=7,
            color=color,
            alpha=0.18,
            linewidths=0,
            zorder=3)
    threshold = microglia_overlap_fraction_threshold(object_table)
    if threshold is not None:
        ax.axhline(threshold, color="#4c6070", linestyle="--", linewidth=1.0, zorder=2)
        ax.text(
            0.02,
            threshold,
            f" threshold={threshold:g}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7.8,
            color="#4c6070")
    ax.set_xticks([1, 2], [f"CTX\n(n={len(data[0])})", f"CA1\n(n={len(data[1])})"])
    apply_axes_controls(ax, config)
    finalize_panel(fig, output_path, config)

def draw_microglia_result_panels(
    manifest: pd.DataFrame,
    view_config: MicrogliaFigureViewConfig = MICROGLIA_VIEW_CONFIG,
) -> None:
    aggregated_manifest = aggregate_microglia_manifest_by_mouse_region(manifest)
    mouse_treatment_manifest = aggregate_microglia_manifest_by_mouse_treatment(manifest)
    paired_manifest = aggregate_paired_microglia_manifest(manifest)
    object_table = load_microglia_cell_object_table(manifest)
    stats_map = compute_microglia_statistics(
        paired_manifest,
        apply_holm_correction=view_config.apply_holm_correction,
    )
    representative_ctx = choose_microglia_row_by_stack_name(manifest, view_config.ctx_stack_name)
    representative_ca1 = choose_microglia_row_by_stack_name(manifest, view_config.ca1_stack_name)

    ctx_results_dir = Path(representative_ctx["results_dir"])
    ctx_source_path = Path(representative_ctx["source_path"])
    ctx_stem = Path(str(representative_ctx["stack_name"])).stem
    ca1_results_dir = Path(representative_ca1["results_dir"])
    ca1_source_path = Path(representative_ca1["source_path"])
    ca1_stem = Path(str(representative_ca1["stack_name"])).stem

    preview_map = {
        "ctx": load_microglia_channel_views(ctx_source_path, view_config),
        "ca1": load_microglia_channel_views(ca1_source_path, view_config)}
    mask_map = {
        "ctx": load_microglia_mask_views(ctx_results_dir, ctx_stem, view_config),
        "ca1": load_microglia_mask_views(ca1_results_dir, ca1_stem, view_config)}

    for region_key, (region_preview_map, microns_per_pixel) in preview_map.items():
        for suffix_key, image in region_preview_map.items():
            panel_key = f"{region_key}_{suffix_key}"
            save_image_panel(
                image,
                MICROGLIA_PANELS[panel_key],
                output_path_for(MICROGLIA_PANELS[panel_key]),
                microns_per_pixel=microns_per_pixel)
            save_image_panel(
                center_crop_image(image, view_config.zoom_crop_size_px),
                MICROGLIA_ZOOM_PANELS[panel_key],
                output_path_for(MICROGLIA_ZOOM_PANELS[panel_key]),
                microns_per_pixel=microns_per_pixel)

    for region_key, region_mask_map in mask_map.items():
        for suffix_key, mask in region_mask_map.items():
            panel_key = f"{region_key}_{suffix_key}"
            save_label_mask_panel(mask, MICROGLIA_PANELS[panel_key], output_path_for(MICROGLIA_PANELS[panel_key]))
            save_label_mask_panel(
                center_crop_image(mask, view_config.zoom_crop_size_px),
                MICROGLIA_ZOOM_PANELS[panel_key],
                output_path_for(MICROGLIA_ZOOM_PANELS[panel_key]))

    metric_panel_map = {
        "n_cells": "n_cells_per_mm3",
        "n_iba1_positive_cells": "n_iba1_positive_cells_per_mm3",
        "iba1_positive_fraction": "iba1_positive_fraction",
        "mean_cell_area": "mean_cell_area_um2_2d",
        "mean_cell_roundness": "mean_cell_roundness_2d",
        "mean_cell_eccentricity": "mean_cell_eccentricity_2d"}
    for panel_key, metric_column in metric_panel_map.items():
        draw_group_comparison_panel(
            manifest=paired_manifest,
            metric_column=metric_column,
            stats_map=stats_map,
            config=MICROGLIA_PANELS[panel_key],
            output_path=output_path_for(MICROGLIA_PANELS[panel_key]))
        draw_treatment_region_panel(
            manifest=aggregated_manifest,
            metric_column=metric_column,
            config=MICROGLIA_TREATMENT_PANELS[panel_key],
            output_path=output_path_for(MICROGLIA_TREATMENT_PANELS[panel_key]))
        draw_global_treatment_panel(
            manifest=mouse_treatment_manifest,
            metric_column=metric_column,
            config=MICROGLIA_TREATMENT_GLOBAL_PANELS[panel_key],
            output_path=output_path_for(MICROGLIA_TREATMENT_GLOBAL_PANELS[panel_key]))

    draw_group_comparison_panel(
        manifest=paired_manifest,
        metric_column="mean_cell_brightness",
        stats_map=stats_map,
        config=MICROGLIA_PANELS["mean_cell_brightness"],
        output_path=output_path_for(MICROGLIA_PANELS["mean_cell_brightness"]))
    draw_global_treatment_panel(
        manifest=mouse_treatment_manifest,
        metric_column="mean_cell_brightness",
        config=MICROGLIA_TREATMENT_GLOBAL_PANELS["mean_cell_brightness"],
        output_path=output_path_for(MICROGLIA_TREATMENT_GLOBAL_PANELS["mean_cell_brightness"]))
    draw_morphology_effect_raw_panel(
        manifest=paired_manifest,
        config=MICROGLIA_PANELS["morphology_effect_raw"],
        output_path=output_path_for(MICROGLIA_PANELS["morphology_effect_raw"]))
    draw_morphology_effect_standardized_panel(
        manifest=paired_manifest,
        config=MICROGLIA_PANELS["morphology_effect_standardized"],
        output_path=output_path_for(MICROGLIA_PANELS["morphology_effect_standardized"]))
    draw_microglia_qc_density_positive_fraction_panel(
        manifest=aggregated_manifest,
        config=MICROGLIA_PANELS["qc_density_positive_fraction"],
        output_path=output_path_for(MICROGLIA_PANELS["qc_density_positive_fraction"]))
    draw_cell_size_distribution_violin_panel(
        object_table=object_table,
        config=MICROGLIA_PANELS["cell_size_distribution_violin"],
        output_path=output_path_for(MICROGLIA_PANELS["cell_size_distribution_violin"]))
    draw_cell_size_distribution_strip_panel(
        object_table=object_table,
        config=MICROGLIA_PANELS["cell_size_distribution_strip"],
        output_path=output_path_for(MICROGLIA_PANELS["cell_size_distribution_strip"]))
    draw_cell_size_distribution_histogram_panel(
        object_table=object_table,
        config=MICROGLIA_PANELS["cell_size_distribution_histogram"],
        output_path=output_path_for(MICROGLIA_PANELS["cell_size_distribution_histogram"]))
    draw_area_roundness_scatter_panel(
        object_table=object_table,
        config=MICROGLIA_PANELS["area_roundness_scatter"],
        output_path=output_path_for(MICROGLIA_PANELS["area_roundness_scatter"]))
    draw_count_median_area_scatter_panel(
        manifest=manifest,
        object_table=object_table,
        config=MICROGLIA_PANELS["count_median_area_scatter"],
        output_path=output_path_for(MICROGLIA_PANELS["count_median_area_scatter"]))
    draw_iba1_overlap_fraction_histogram_panel(
        object_table=object_table,
        config=MICROGLIA_PANELS["iba1_overlap_fraction_histogram"],
        output_path=output_path_for(MICROGLIA_PANELS["iba1_overlap_fraction_histogram"]))
    draw_iba1_overlap_fraction_distribution_panel(
        object_table=object_table,
        config=MICROGLIA_PANELS["iba1_overlap_fraction_distribution"],
        output_path=output_path_for(MICROGLIA_PANELS["iba1_overlap_fraction_distribution"]))

def synthetic_results_available(paths: SyntheticBenchmarkPaths) -> bool:
    expected = paths.results_dir / f"{synthetic_result_stem('synthetic_stack_00')}_roi_overview.csv"
    return expected.exists()
# %% MAIN FUNCTION
def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        })

    if synthetic_results_available(SYNTHETIC_SHARP_PATHS):
        synthetic_evaluation = load_synthetic_evaluation(SYNTHETIC_SHARP_PATHS)
        draw_synthetic_benchmark_panels(synthetic_evaluation, SYNTHETIC_PANELS, paths=SYNTHETIC_SHARP_PATHS)
    else:
        print(
            "Skipping main synthetic benchmark panels because no sharp synthetic results were found in "
            f"{SYNTHETIC_SHARP_PATHS.results_dir}.")
    if synthetic_results_available(SYNTHETIC_GAUSSIAN_PATHS):
        synthetic_gaussian_evaluation = load_synthetic_evaluation(SYNTHETIC_GAUSSIAN_PATHS)
        draw_synthetic_benchmark_panels(synthetic_gaussian_evaluation, SYNTHETIC_SUPPLEMENT_PANELS,
                                        paths=SYNTHETIC_GAUSSIAN_PATHS)
    else:
        print("Skipping supplementary synthetic benchmark panels because no Gaussian synthetic results were found in "
              f"{SYNTHETIC_GAUSSIAN_PATHS.results_dir}.")
    if synthetic_results_available(SYNTHETIC_GAUSSIAN_CELLPOSE_PATHS):
        synthetic_gaussian_cellpose_evaluation = load_synthetic_evaluation(SYNTHETIC_GAUSSIAN_CELLPOSE_PATHS)
        draw_synthetic_benchmark_panels(synthetic_gaussian_cellpose_evaluation, SYNTHETIC_GAUSSIAN_CELLPOSE_PANELS,
                                        paths=SYNTHETIC_GAUSSIAN_CELLPOSE_PATHS)
    else:
        print("Skipping Gaussian Cellpose synthetic benchmark panels because no Gaussian Cellpose synthetic results "
              f"were found in {SYNTHETIC_GAUSSIAN_CELLPOSE_PATHS.results_dir}.")
    if synthetic_results_available(SYNTHETIC_SHARP_CELLPOSE_PATHS):
        synthetic_cellpose_evaluation = load_synthetic_evaluation(SYNTHETIC_SHARP_CELLPOSE_PATHS)
        draw_synthetic_benchmark_panels(synthetic_cellpose_evaluation, SYNTHETIC_CELLPOSE_SUPPLEMENT_PANELS,
                                        paths=SYNTHETIC_SHARP_CELLPOSE_PATHS)
    else:
        print("Skipping Cellpose synthetic benchmark panels because no sharp Cellpose synthetic results were found in "
              f"{SYNTHETIC_SHARP_CELLPOSE_PATHS.results_dir}.")
    microglia_manifest = load_microglia_manifest()
    draw_microglia_result_panels(microglia_manifest)
# %% MAIN
if __name__ == "__main__":
    main()
# %% END
