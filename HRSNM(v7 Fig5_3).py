# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:58:57 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
Figure 5 plotting + data export
================================
Purpose:
  1) Build the global catchment dataset (same processing pipeline as before).
  2) Plot Figure 5 (9-panel main figure: a map, b/c/d bars, e/f/g pies,
     and h/i relationship plots).
  3) Export two CSVs used by the separate Results-text script:
        - figure5_catchment_data_<TAG>_OUTLIER_<ON/OFF>.csv   (per-catchment)
        - figure5_subregion_agg_<TAG>_OUTLIER_<ON/OFF>.csv     (per-subregion aggregates)

@author: zouxu
"""

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Global plotting setup
# ============================================================
mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "hatch.linewidth": 0.35,
    "axes.formatter.useoffset": False,
})

# ============================================================
# Configuration
# ============================================================
TAG = "T_20_SO4_15_COD_525"
CORROSION_RATE_THRESHOLD_MM_PER_YEAR = 1.0
MIN_SERVICE_POPULATION_RATIO = 0.1
MAX_SERVICE_POPULATION_RATIO = 10.0
MIN_SUBREGION_CATCHMENTS = 200

input_code2_csv = os.environ.get(
    "HRSNM_FIG5_GLOBAL_CSV",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "global",
        "catchment_summary.csv",
    )),
)

input_reg_csv = os.environ.get(
    "HRSNM_FIG5_REGRESSION_CSV",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fig4_results_export",
        f"regressed_equations_{TAG}.csv",
    ),
)

COUNTRIES_SHP = os.environ.get(
    "HRSNM_FIG5_COUNTRIES_SHP",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "global",
        "ne_10m_admin_0_countries", "ne_10m_admin_0_countries.shp",
    ),
)

output_dir = os.path.abspath(os.environ.get(
    "HRSNM_FIG5_OUT_DIR",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fig5_results_export",
    ),
))
os.makedirs(output_dir, exist_ok=True)
NO_SHOW = os.environ.get("HRSNM_FIG5_NO_SHOW", "0") == "1"

# ============================================================
# Figure 5 pie / title alignment settings
# ============================================================
FIG1_PIE_RADIUS = 1.5
FIG1_PIE_LABEL_REGIONS = ["Northern America", "Eastern Asia"]
FIG1_PIE_TITLE_Y = 1.02
FIG1_BAR_TITLE_Y = 1.00

# ============================================================
# Outlier filtering switches
# ============================================================
REMOVE_OUTLIERS = False

FILTER_SUM_L_B_OUTLIERS = True
SUM_L_B_QUANTILE_LOW = 0.01
SUM_L_B_QUANTILE_HIGH = 0.99
SUM_L_B_ABS_MAX = None

FILTER_L_OVER_Q_OUTLIERS = True
L_OVER_Q_QUANTILE_LOW = 0.01
L_OVER_Q_QUANTILE_HIGH = 0.99
L_OVER_Q_ABS_MAX = None

DROP_FILTERED_ROWS = True

if not REMOVE_OUTLIERS:
    FILTER_SUM_L_B_OUTLIERS = False
    FILTER_L_OVER_Q_OUTLIERS = False
    DROP_FILTERED_ROWS = False
    print("\n[Outlier filter] >>> REMOVE_OUTLIERS = False : ALL outlier filtering DISABLED. <<<\n")
else:
    print("\n[Outlier filter] >>> REMOVE_OUTLIERS = True : outlier filtering ENABLED. <<<\n")

_suffix = "ON" if REMOVE_OUTLIERS else "OFF"

FIGURE_H_LOG_SCALE = True

# ============================================================
# Figure h linear-axis scaling
# ============================================================
FIG1_H_X_SCALE = 1e9
FIG1_H_Y_SCALE = 1e7
FIG1_H_XLIM_SCALED = (0, 1.2)
FIG1_H_YLIM_SCALED = (0, 3.5)

# ============================================================
# Scatter size config
# ============================================================
SCATTER_SIZE_MODE = "linear"
SCATTER_SIZE_GAIN = 1.0
SCATTER_SIZE_MIN_DEFAULT = 1
SCATTER_SIZE_MAX_DEFAULT = 10.0

SCATTER_PANEL_CONFIG = {
    "a": {"vmin": None, "vmax": None, "smin": SCATTER_SIZE_MIN_DEFAULT, "smax": SCATTER_SIZE_MAX_DEFAULT, "gain": SCATTER_SIZE_GAIN},
    "c": {"vmin": None, "vmax": None, "smin": SCATTER_SIZE_MIN_DEFAULT, "smax": SCATTER_SIZE_MAX_DEFAULT, "gain": SCATTER_SIZE_GAIN},
    "e": {"vmin": None, "vmax": None, "smin": SCATTER_SIZE_MIN_DEFAULT, "smax": SCATTER_SIZE_MAX_DEFAULT, "gain": SCATTER_SIZE_GAIN},
    "g": {"vmin": None, "vmax": None, "smin": SCATTER_SIZE_MIN_DEFAULT, "smax": SCATTER_SIZE_MAX_DEFAULT, "gain": SCATTER_SIZE_GAIN},
}


def get_panel_size_config(panel_label):
    cfg = SCATTER_PANEL_CONFIG.get(panel_label, {})
    smin = cfg.get("smin") or SCATTER_SIZE_MIN_DEFAULT
    smax = cfg.get("smax") or SCATTER_SIZE_MAX_DEFAULT
    return dict(
        size_mode=SCATTER_SIZE_MODE,
        size_gain=SCATTER_SIZE_GAIN * cfg.get("gain", 1.0),
        size_vmin=cfg.get("vmin"),
        size_vmax=cfg.get("vmax"),
        size_min=smin,
        size_max=smax,
    )


def compute_value_sizes(values, mode="linear", vmin=None, vmax=None,
                        smin=1.0, smax=30.0, gain=1.0):
    arr = np.asarray(values, dtype=float)
    sizes = np.full(arr.shape, smin, dtype=float)
    mask = np.isfinite(arr) & (arr > 0)
    if mask.sum() == 0:
        return sizes * gain
    valid = arr[mask]
    if vmin is None or not np.isfinite(vmin):
        vmin = float(np.nanpercentile(valid, 2))
        if not np.isfinite(vmin) or vmin <= 0:
            vmin = float(np.nanmin(valid))
    if vmax is None or not np.isfinite(vmax):
        vmax = float(np.nanpercentile(valid, 98))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = float(np.nanmax(valid))
    if vmax <= vmin:
        sizes[mask] = 0.5 * (smin + smax)
        return sizes * gain
    if mode == "exp":
        vmin_pos = max(vmin, 1e-30)
        vmax_pos = max(vmax, vmin_pos * 10)
        v_clip = np.clip(valid, vmin_pos, vmax_pos)
        scaled = (np.log10(v_clip) - np.log10(vmin_pos)) / (np.log10(vmax_pos) - np.log10(vmin_pos))
    else:
        v_clip = np.clip(valid, vmin, vmax)
        scaled = (v_clip - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    sizes[mask] = smin + scaled * (smax - smin)
    return sizes * gain


KM2_TO_M2 = 1e6

COLORS = {
    "text": "#222222", "subtle_text": "#555555", "grid": "#D9D9D9",
    "spine": "#333333", "map_land": "#F2F2F2", "map_edge": "#CFCFCF",
    "nodata": "#A8A8A8", "note": "#666666", "box_edge": "#333333",
    "soft_green": "#A1B4AC", "soft_beige": "#E7E3DA", "soft_brown": "#A66842",
    "deep_brown": "#9C3106", "excluded_face": "#E4E4E4", "excluded_edge": "#9C9C9C",
    "excluded_hatch": "#7A7A7A", "qualified_country": "#EAEAEA",
}

CMAP_PANEL_D = LinearSegmentedColormap.from_list("panel_d_cmap",
    ["#E0EAE6", "#BFD3CB", "#A1B4AC", "#82BFB0", "#5FA493"])
CMAP_PANEL_E = LinearSegmentedColormap.from_list("panel_e_cmap",
    ["#A5D2D3", "#8CC5C6", "#65B1B3", "#4D9B9D", "#3E7C7E"])
CMAP_PANEL_F = LinearSegmentedColormap.from_list("panel_f_cmap",
    ["#C2D6DE", "#8FB3C2", "#6F9FB5", "#3A7A8C", "#1B4F66"])
CMAP_LQ = LinearSegmentedColormap.from_list("lq_cmap",
    ["#F1E3DF", "#B08682", "#674541"])

REGION_PALETTE = [
    "#82BFB0", "#3A7A8C", "#6F9FB5", "#A66842",
    "#9C6B4E", "#A1B4AC", "#C7B9A5", "#8A8A8A",
]

EXCLUDED_HATCH_PATTERN = "////"

SUBREGION_ABBR = {
    "Central America": "C. America", "Eastern Asia": "E. Asia",
    "Northern Africa": "N. Africa", "Western Asia": "W. Asia",
    "Northern America": "N. America", "Southern Africa": "S. Africa",
    "South-Eastern Asia": "SE Asia", "Australia and New Zealand": "Aus. & NZ",
    "South America": "S. America", "Eastern Europe": "E. Europe",
    "Western Europe": "W. Europe", "Northern Europe": "N. Europe",
    "Southern Europe": "S. Europe",
}


def abbr(name):
    if pd.isna(name):
        return name
    return SUBREGION_ABBR.get(str(name), str(name))


# ============================================================
# Helpers
# ============================================================
def powerlaw_predict(x, slope, intercept_log10):
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(x) & (x > 0)
    y[mask] = 10 ** (intercept_log10 + slope * np.log10(x[mask]))
    return y


def get_regression_params(reg_df, y_var):
    sub = reg_df[reg_df["y_var"] == y_var].copy()
    if sub.empty:
        raise ValueError(f"Cannot find regression for y_var = {y_var}")
    row = sub.iloc[0]
    return float(row["slope_log10"]), float(row["intercept_log10"])


def robust_positive_limits(values, qlow=2, qhigh=98):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return 1e-6, 1.0
    vmin = np.percentile(arr, qlow)
    vmax = np.percentile(arr, qhigh)
    if not np.isfinite(vmin) or vmin <= 0:
        vmin = arr.min()
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = arr.max()
    if vmax <= vmin:
        vmax = vmin * 10
    return vmin, vmax


def robust_limits_any(values, qlow=2, qhigh=98):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 1.0
    vmin = np.percentile(arr, qlow)
    vmax = np.percentile(arr, qhigh)
    if not np.isfinite(vmin):
        vmin = arr.min()
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = arr.max()
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def clean_axis(ax, keep_left=False, keep_bottom=False):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_visible(keep_bottom)
    if keep_left:
        ax.spines["left"].set_color(COLORS["spine"])
        ax.spines["left"].set_linewidth(0.6)
    if keep_bottom:
        ax.spines["bottom"].set_color(COLORS["spine"])
        ax.spines["bottom"].set_linewidth(0.6)


def add_panel_label(ax, label, title, title_y=None):
    txt = f"{label}" if (title is None or str(title).strip() == "") else f"{label}  {title}"
    kwargs = dict(label=txt, loc="left", pad=4, fontsize=9,
                  fontweight="bold", color=COLORS["text"])
    if title_y is not None:
        kwargs["y"] = title_y
    ax.set_title(**kwargs)


def format_log_colorbar(cb):
    cb.ax.tick_params(labelsize=7, width=0.5, length=2, colors=COLORS["subtle_text"])
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(COLORS["spine"])


def detect_sum_L_b_column(df):
    candidate_names = ["sum_L_b", "sum(L_b)", "sum_Lb", "L_b", "Lb", "total_L_b"]
    for c in candidate_names:
        if c in df.columns:
            return c
    normalized = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "").replace("(", "").replace(")", "").replace("-", "_")
        normalized[c] = key
    for original, key in normalized.items():
        if key in ["sum_l_b", "sumlb", "lb", "l_b"]:
            return original
    return None


def detect_sum_L_m_column(df):
    candidate_names = ["sum_L_m", "sum(L_m)", "sum_Lm", "sum_L", "L_m",
                       "total_L_m", "sum_pipe_length_m", "pipe_length_m"]
    for c in candidate_names:
        if c in df.columns:
            return c
    normalized = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "").replace("(", "").replace(")", "").replace("-", "_")
        normalized[c] = key
    for original, key in normalized.items():
        if key in ["sum_l_m", "sumlm", "l_m", "lm", "sum_l"]:
            return original
    for original, key in normalized.items():
        if "sum" in key and "l" in key and ("_m" in key or key.endswith("m")):
            if "lb" not in key and "l_b" not in key:
                return original
    return None


def build_positive_quantile_filter(series, qlow=0.01, qhigh=0.99, abs_max=None):
    s = pd.to_numeric(series, errors="coerce")
    valid = np.isfinite(s) & (s > 0)
    if valid.sum() == 0:
        return valid, np.nan, np.nan
    positive_values = s[valid].astype(float)
    lower = positive_values.min() if qlow is None else positive_values.quantile(qlow)
    upper = positive_values.max() if qhigh is None else positive_values.quantile(qhigh)
    if not np.isfinite(lower) or lower <= 0:
        lower = positive_values.min()
    if not np.isfinite(upper) or upper <= lower:
        upper = positive_values.max()
    if abs_max is not None and np.isfinite(abs_max) and abs_max > 0:
        upper = min(upper, abs_max)
    mask_keep = valid & (s >= lower) & (s <= upper)
    return mask_keep, float(lower), float(upper)


def _plot_excluded_layer(ax, excluded_world):
    if excluded_world is None or len(excluded_world) == 0:
        return
    mpl.rcParams["hatch.color"] = COLORS["excluded_hatch"]
    excluded_world.plot(ax=ax, facecolor=COLORS["excluded_face"],
                        edgecolor=COLORS["excluded_edge"], linewidth=0.25,
                        hatch=EXCLUDED_HATCH_PATTERN, zorder=1)


def _add_excluded_legend(ax, loc="lower left"):
    patch = Patch(facecolor=COLORS["excluded_face"], edgecolor=COLORS["excluded_edge"],
                  hatch=EXCLUDED_HATCH_PATTERN, linewidth=0.4,
                  label="Not considered\n(no matched catchment data)")
    leg = ax.legend(handles=[patch], loc=loc, bbox_to_anchor=(0.01, 0.01),
                    frameon=True, framealpha=0.92, edgecolor=COLORS["map_edge"],
                    fontsize=6.5, handlelength=1.6, handleheight=1.2,
                    borderpad=0.35, labelspacing=0.3)
    leg.get_frame().set_linewidth(0.4)
    return leg


def plot_map_panel(ax, world, df, value_col, panel_label, panel_title,
                   cax=None, cbar_label="", cmap="viridis", vmin=None, vmax=None,
                   size_min=1.0, size_max=30.0, size_mode="linear", size_gain=1.0,
                   size_vmin=None, size_vmax=None, excluded_world=None,
                   show_excluded_legend=False, marker="o"):
    ax.set_facecolor(COLORS["map_land"])
    world.plot(ax=ax, color=COLORS["qualified_country"],
               edgecolor=COLORS["map_edge"], linewidth=0.25, zorder=0)
    _plot_excluded_layer(ax, excluded_world)

    sub = df[np.isfinite(df["LON_WWTP"]) & np.isfinite(df["LAT_WWTP"])].copy()
    sub = sub[np.isfinite(sub[value_col]) & (sub[value_col] > 0)].copy()

    add_panel_label(ax, panel_label, panel_title)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 85)
    ax.set_xticks([-120, -60, 0, 60, 120]); ax.set_yticks([-40, 0, 40, 80])
    ax.tick_params(colors=COLORS["subtle_text"], length=2.5, width=0.6)
    ax.set_xlabel(""); ax.set_ylabel("")
    clean_axis(ax, keep_left=True, keep_bottom=True)

    if sub.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center",
                va="center", fontsize=8, color=COLORS["nodata"])
        if cax is not None:
            cax.axis("off")
    else:
        if vmin is None or vmax is None:
            vmin, vmax = robust_positive_limits(sub[value_col].values, 2, 98)
        sizes = compute_value_sizes(sub[value_col].values, mode=size_mode,
                                    vmin=size_vmin, vmax=size_vmax,
                                    smin=size_min, smax=size_max, gain=size_gain)
        sc = ax.scatter(sub["LON_WWTP"], sub["LAT_WWTP"], c=sub[value_col], s=sizes,
                        cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), alpha=0.95,
                        edgecolors="white", linewidths=0.1, zorder=3, marker=marker)
        if cax is not None:
            cb = plt.colorbar(sc, cax=cax, orientation="vertical")
            cb.set_label(cbar_label, fontsize=8, color=COLORS["text"], labelpad=4)
            format_log_colorbar(cb)

    if show_excluded_legend:
        _add_excluded_legend(ax)


def plot_relationship_scatter(ax, df, x_col="sum_L_m", y_col="sum_L_over_Q",
                              region_col="SUBREGION", panel_label="h",
                              panel_title="Relationship of catchment metrics",
                              region_colors=None, use_log=True, error_type="sem",
                              center_type="mean", linear_x_scale=1.0, linear_y_scale=1.0,
                              linear_xlim_scaled=None, linear_ylim_scaled=None,
                              x_label_log=None, y_label_log=None,
                              x_label_linear=None, y_label_linear=None,
                              size_col=None, size_label=None,
                              region_legend_loc="lower center",
                              region_legend_bbox=(0.5, 0.01),
                              region_legend_ncol=4,
                              region_legend_frame=True,
                              region_legend_title="SUBREGION",
                              include_size_in_region_legend=False,
                              show_region_legend=True,
                              size_legend_loc="upper left",
                              show_size_legend=False,
                              annotate_region_sizes=True):
    add_panel_label(ax, panel_label, panel_title)

    cols_needed = [x_col, y_col]
    if size_col is not None and size_col not in cols_needed:
        cols_needed.append(size_col)
    has_region = (region_col in df.columns)
    if has_region:
        cols_needed = cols_needed + [region_col]

    sub = df[cols_needed].copy()
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    sub[y_col] = pd.to_numeric(sub[y_col], errors="coerce")
    if size_col is not None:
        sub[size_col] = pd.to_numeric(sub[size_col], errors="coerce")
    sub = sub.dropna(subset=[x_col, y_col])
    if use_log:
        sub = sub[(sub[x_col] > 0) & (sub[y_col] > 0)]
    if size_col is not None:
        sub = sub[np.isfinite(sub[size_col]) & (sub[size_col] > 0)].copy()

    if sub.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center",
                va="center", fontsize=8, color=COLORS["nodata"])
        clean_axis(ax, keep_left=True, keep_bottom=True)
        return

    px, py = "__plot_x__", "__plot_y__"
    if use_log:
        sub[px] = sub[x_col]; sub[py] = sub[y_col]
    else:
        sub[px] = sub[x_col] / float(linear_x_scale)
        sub[py] = sub[y_col] / float(linear_y_scale)

    point_sizes = None
    size_vmin = None
    size_vmax = None
    if size_col is not None:
        size_values = sub[size_col].to_numpy(dtype=float)
        size_vmin, size_vmax = robust_positive_limits(size_values, 2, 98)
        point_sizes = compute_value_sizes(
            size_values, mode="log", vmin=size_vmin, vmax=size_vmax,
            smin=1.0, smax=16.0, gain=1.0,
        )

    def _center_err(values):
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        n = len(v)
        if n == 0:
            return np.nan, 0.0, 0.0
        c = np.median(v) if center_type == "median" else np.mean(v)
        if n < 2:
            return c, 0.0, 0.0
        if error_type == "sem":
            e = np.std(v, ddof=1) / np.sqrt(n); return c, e, e
        elif error_type == "ci95":
            e = 1.96 * np.std(v, ddof=1) / np.sqrt(n); return c, e, e
        elif error_type == "sd":
            sd = np.std(v, ddof=1); return c, sd, sd
        elif error_type == "iqr":
            q1 = np.percentile(v, 25); q3 = np.percentile(v, 75)
            return c, max(c - q1, 0.0), max(q3 - c, 0.0)
        elif error_type == "mad":
            mad = np.median(np.abs(v - np.median(v))); return c, mad, mad
        e = np.std(v, ddof=1) / np.sqrt(n); return c, e, e

    if has_region and region_colors is not None:
        regions_in_data = sorted(sub[region_col].dropna().unique().tolist(), key=lambda r: abbr(r))
        legend_handles = []
        representatives = []
        for r in regions_in_data:
            mask = sub[region_col] == r
            if mask.sum() == 0:
                continue
            color = region_colors.get(r, "#999999")
            scatter_sizes = (
                point_sizes[mask.to_numpy()] if point_sizes is not None else 2.5
            )
            ax.scatter(sub.loc[mask, px], sub.loc[mask, py], c=color, s=scatter_sizes,
                       alpha=0.2, linewidths=0.0, edgecolors="none", zorder=3)
        for r in regions_in_data:
            mask = sub[region_col] == r
            if mask.sum() < 2:
                continue
            x_c, x_lo, x_hi = _center_err(sub.loc[mask, px].values.astype(float))
            y_c, y_lo, y_hi = _center_err(sub.loc[mask, py].values.astype(float))
            if not (np.isfinite(x_c) and np.isfinite(y_c)):
                continue
            if use_log:
                x_lo = min(x_lo, max(x_c * 0.999, 0.0))
                y_lo = min(y_lo, max(y_c * 0.999, 0.0))
            color = region_colors.get(r, "#999999")
            size_c = (
                float(np.nanmean(sub.loc[mask, size_col]))
                if size_col is not None else np.nan
            )
            marker_area = (
                compute_value_sizes(
                    np.array([size_c]), mode="log",
                    vmin=size_vmin, vmax=size_vmax,
                    smin=22.0, smax=95.0, gain=1.0,
                )[0]
                if np.isfinite(size_c) and size_c > 0 else 25.0
            )
            marker_size = float(np.sqrt(marker_area))
            ax.errorbar(x_c, y_c, xerr=np.array([[x_lo], [x_hi]]),
                        yerr=np.array([[y_lo], [y_hi]]), fmt="o", color=color,
                        ecolor=color, elinewidth=1.0, capsize=2.5, capthick=1.0,
                        markersize=marker_size, markerfacecolor=color, markeredgecolor="white",
                        markeredgewidth=0.6, alpha=0.95, zorder=6)
            representatives.append((r, x_c, y_c, size_c, color, marker_size))
            legend_handles.append(Line2D(
                [0], [0], marker="o", linestyle="", color=color,
                markersize=4.5, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.45,
                label=abbr(r),
            ))
        err_label_map = {"sem": "Mean ± SEM", "ci95": "Mean ± 95% CI",
                         "sd": "Mean ± SD", "iqr": "Median (IQR)", "mad": "Median ± MAD"}
        err_label = err_label_map.get(error_type, "Mean ± SEM")
        legend_handles.append(Line2D(
            [0], [0], marker="o", linestyle="-", color="#444444",
            markersize=5.0, markerfacecolor="#444444",
            markeredgecolor="white", markeredgewidth=0.6, label=err_label,
        ))
        size_handle = None
        if size_col is not None and size_label:
            size_handle = Line2D(
                [0], [0], marker="o", linestyle="", color="#666666",
                markersize=7.0, markerfacecolor="#999999",
                markeredgecolor="white", markeredgewidth=0.45,
                label=f"Marker size: {size_label}",
            )
        if include_size_in_region_legend and size_handle is not None:
            legend_handles.append(size_handle)
        if legend_handles and show_region_legend:
            leg = ax.legend(
                handles=legend_handles, loc=region_legend_loc,
                ncol=region_legend_ncol,
                bbox_to_anchor=region_legend_bbox,
                frameon=region_legend_frame, framealpha=0.90, fontsize=4.9,
                handlelength=0.8, handleheight=0.8, columnspacing=0.55,
                borderpad=0.25, labelspacing=0.14,
                title=region_legend_title,
                title_fontsize=5.5,
            )
            leg._legend_box.align = "left"
            leg.set_gid("figure5_region_legend")
            if show_size_legend:
                ax.add_artist(leg)
        if size_handle is not None and show_size_legend:
            size_leg = ax.legend(
                handles=[size_handle], loc=size_legend_loc,
                frameon=False, framealpha=0.90, fontsize=5.4,
                handlelength=0.9, borderpad=0.28, labelspacing=0.18,
            )
            size_leg._legend_box.align = "left"
    else:
        ax.scatter(sub[px], sub[py], c="#3A7A8C", s=7, alpha=0.7, linewidths=0)

    if use_log:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, which="major", color=COLORS["grid"], linestyle=":", linewidth=0.4, alpha=0.6)
        ax.set_xlabel(
            x_label_log or r"$\sum L_b$ (m)",
            fontsize=8, color=COLORS["text"],
        )
        ax.set_ylabel(
            y_label_log or r"$\sum L_b/Q_b$ (m$\cdot$yr/m$^3$)",
            fontsize=8, color=COLORS["text"],
        )
    else:
        ax.set_xscale("linear"); ax.set_yscale("linear")
        if linear_xlim_scaled is not None:
            ax.set_xlim(*linear_xlim_scaled)
        if linear_ylim_scaled is not None:
            ax.set_ylim(*linear_ylim_scaled)
        ax.grid(True, which="major", color=COLORS["grid"], linestyle=":", linewidth=0.4, alpha=0.6)
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.set_xlabel(
            x_label_linear or r"$\sum L_b$ ($10^9$ m)",
            fontsize=8, color=COLORS["text"],
        )
        ax.set_ylabel(
            y_label_linear
            or r"$\sum L_b/Q_b$ ($10^7$ m$\cdot$yr/m$^3$)",
            fontsize=8, color=COLORS["text"],
        )

    ax.tick_params(axis="both", labelsize=7, colors=COLORS["subtle_text"])
    clean_axis(ax, keep_left=True, keep_bottom=True)

    if (
        has_region and region_colors is not None
        and size_col is not None and annotate_region_sizes
        and representatives
    ):
        valid_reps = [
            rep for rep in representatives
            if np.isfinite(rep[1]) and np.isfinite(rep[2])
            and np.isfinite(rep[3]) and rep[3] > 0
        ]
        if valid_reps:
            if show_region_legend or show_size_legend:
                label_groups = [(valid_reps, 0.97, "right")]
            else:
                log_x = np.log10([rep[1] for rep in valid_reps])
                median_log_x = float(np.median(log_x))
                left_reps = [
                    rep for rep in valid_reps
                    if np.log10(rep[1]) <= median_log_x
                ]
                right_reps = [
                    rep for rep in valid_reps
                    if np.log10(rep[1]) > median_log_x
                ]
                label_groups = [
                    (left_reps, 0.03, "left"),
                    (right_reps, 0.97, "right"),
                ]
            for reps, x_text, ha in label_groups:
                reps = sorted(reps, key=lambda rep: rep[2])
                if not reps:
                    continue
                y_start = 0.28 if show_region_legend else 0.08
                y_positions = np.linspace(y_start, 0.76, len(reps))
                for rep, y_text in zip(reps, y_positions):
                    _, x_c, y_c, size_c, color, _ = rep
                    ax.annotate(
                        f"{size_c:.1e}",
                        xy=(x_c, y_c), xycoords="data",
                        xytext=(x_text, y_text), textcoords=ax.transAxes,
                        ha=ha, va="center", fontsize=4.8, color=color,
                        bbox=dict(
                            boxstyle="round,pad=0.10", facecolor="white",
                            edgecolor="none", alpha=0.78,
                        ),
                        arrowprops=dict(
                            arrowstyle="-", color=color, linewidth=0.35,
                            alpha=0.75, shrinkA=1.5, shrinkB=2.0,
                        ),
                        annotation_clip=True, zorder=8,
                    )


def format_total_value(value, unit="", decimals=0):
    if not np.isfinite(value):
        return "NA"
    txt = f"{value:,.{decimals}f}"
    return f"{txt} {unit}" if unit else txt


def _prepare_region_metric_df(data, y_col, err_col, total_col_for_pie, n_col=None, region_order=None):
    needed_cols = ["SUBREGION", y_col, err_col, total_col_for_pie]
    if n_col is not None and n_col in data.columns:
        needed_cols = needed_cols + [n_col]
    plot_df = data[needed_cols].copy()
    plot_df = plot_df.dropna(subset=["SUBREGION", y_col])
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df[err_col] = pd.to_numeric(plot_df[err_col], errors="coerce")
    plot_df[total_col_for_pie] = pd.to_numeric(plot_df[total_col_for_pie], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df[y_col]) & (plot_df[y_col] > 0)].copy()
    if plot_df.empty:
        return plot_df
    if region_order is None:
        plot_df = plot_df.sort_values(by=y_col, ascending=True).reset_index(drop=True)
    else:
        order_map = {region: i for i, region in enumerate(region_order)}
        plot_df["__order__"] = plot_df["SUBREGION"].map(order_map)
        plot_df = plot_df.dropna(subset=["__order__"]).sort_values("__order__").reset_index(drop=True)
        plot_df = plot_df.drop(columns="__order__")
    plot_df.loc[~np.isfinite(plot_df[err_col]), err_col] = 0.0
    plot_df.loc[plot_df[err_col] < 0, err_col] = 0.0
    return plot_df


def _get_panel_region_color_map(plot_df, region_colors, panel_cmap=None):
    n_bars = len(plot_df)
    if panel_cmap is not None and n_bars > 0:
        fracs = np.array([0.65]) if n_bars == 1 else np.linspace(0.30, 0.92, n_bars)
        bar_colors = [panel_cmap(f) for f in fracs]
        panel_region_color_map = {r: c for r, c in zip(plot_df["SUBREGION"].tolist(), bar_colors)}
    else:
        bar_colors = [region_colors.get(r, "#999999") for r in plot_df["SUBREGION"]]
        panel_region_color_map = dict(region_colors)
    return bar_colors, panel_region_color_map


def plot_region_bar_ordered_with_error(ax, data, y_col, err_col, total_col_for_pie, n_col,
                                       xlabel, panel_label, panel_title, region_colors,
                                       region_order=None, panel_cmap=None, xlim_max=None,
                                       show_n=True):
    add_panel_label(ax, panel_label, panel_title, title_y=FIG1_BAR_TITLE_Y)
    plot_df = _prepare_region_metric_df(data, y_col, err_col, total_col_for_pie, n_col, region_order)
    if plot_df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center",
                va="center", fontsize=8, color=COLORS["nodata"])
        clean_axis(ax, keep_left=False, keep_bottom=True)
        return
    bar_colors, _ = _get_panel_region_color_map(plot_df, region_colors, panel_cmap)
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df[y_col].values, xerr=plot_df[err_col].values, color=bar_colors,
            edgecolor=COLORS["box_edge"], linewidth=0.35, height=0.64, zorder=2,
            error_kw={"elinewidth": 0.65, "ecolor": COLORS["spine"], "capsize": 2.0,
                      "capthick": 0.65, "zorder": 3})
    ax.set_yticks(y_pos)
    ax.set_yticklabels([abbr(r) for r in plot_df["SUBREGION"].values], fontsize=7, color=COLORS["text"])
    max_val = np.nanmax(plot_df[y_col].values + plot_df[err_col].values)
    if not np.isfinite(max_val) or max_val <= 0:
        max_val = np.nanmax(plot_df[y_col].values)
    ref_max = xlim_max if (xlim_max is not None and np.isfinite(xlim_max)) else max_val
    x_offset = ref_max * 0.02 if ref_max > 0 else 0.02
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if show_n and n_col is not None and n_col in plot_df.columns:
            n_val = row.get(n_col, np.nan)
            label_txt = f"n={int(n_val)}" if (pd.notna(n_val) and np.isfinite(n_val) and n_val > 0) else ""
        else:
            label_txt = ""
        ax.text(row[y_col] + row[err_col] + x_offset, i, label_txt, va="center",
                ha="left", fontsize=6.6, color=COLORS["subtle_text"])
    ax.set_xlabel(xlabel, fontsize=8, color=COLORS["text"]); ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=7, colors=COLORS["subtle_text"])
    ax.tick_params(axis="y", labelsize=7, colors=COLORS["text"], length=0)
    ax.grid(False)
    if xlim_max is not None and np.isfinite(xlim_max) and xlim_max > 0:
        ax.set_xlim(0, xlim_max)
    elif max_val > 0:
        ax.set_xlim(0, max_val * 1.55)
    clean_axis(ax, keep_left=False, keep_bottom=True)


def plot_region_share_pie_panel(ax, data, y_col, err_col, total_col_for_pie, n_col,
                                panel_label, panel_title, region_colors, region_order=None,
                                panel_cmap=None, pie_title="Share of total",
                                pie_radius=FIG1_PIE_RADIUS, extra_label_regions=None):
    add_panel_label(ax, panel_label, panel_title, title_y=FIG1_PIE_TITLE_Y)
    if extra_label_regions is None:
        extra_label_regions = FIG1_PIE_LABEL_REGIONS
    plot_df = _prepare_region_metric_df(data, y_col, err_col, total_col_for_pie, n_col, region_order)

    def _empty():
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center",
                va="center", fontsize=8, color=COLORS["nodata"])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if plot_df.empty:
        _empty(); return
    _, panel_region_color_map = _get_panel_region_color_map(plot_df, region_colors, panel_cmap)
    pie_df = plot_df[["SUBREGION", total_col_for_pie]].copy()
    pie_df = pie_df.dropna(subset=["SUBREGION", total_col_for_pie])
    pie_df = pie_df[np.isfinite(pie_df[total_col_for_pie]) & (pie_df[total_col_for_pie] > 0)].copy()
    if pie_df.empty:
        _empty(); return
    total = pie_df[total_col_for_pie].sum()
    if not np.isfinite(total) or total <= 0:
        _empty(); return
    pie_df["share"] = pie_df[total_col_for_pie] / total
    pie_df = pie_df.reset_index(drop=True)
    colors = [panel_region_color_map.get(r, "#999999") for r in pie_df["SUBREGION"]]

    if total_col_for_pie in ["total_problem", "total_toxic"]:
        total_unit = "km"
    elif total_col_for_pie in ["total_area"]:
        total_unit = "m²"
    elif total_col_for_pie in ["total_area_km2"]:
        total_unit = "km²"
    else:
        total_unit = ""
    total_txt = format_total_value(total, unit=total_unit)

    wedges, _texts = ax.pie(pie_df["share"].values, colors=colors, startangle=90,
                            counterclock=False, wedgeprops={"linewidth": 0.25, "edgecolor": "white"},
                            radius=pie_radius)

    label_indices = [int(pie_df["share"].idxmax())]
    for region_name in extra_label_regions:
        matched = pie_df.index[pie_df["SUBREGION"].astype(str) == str(region_name)].tolist()
        if matched:
            label_indices.append(int(matched[0]))
    label_indices = list(dict.fromkeys(label_indices))

    label_entries = []
    for idx in label_indices:
        wedge = wedges[idx]
        share = float(pie_df.loc[idx, "share"])
        region = str(pie_df.loc[idx, "SUBREGION"])
        ang = (wedge.theta2 + wedge.theta1) / 2.0
        ang_rad = np.deg2rad(ang)
        x = np.cos(ang_rad); y = np.sin(ang_rad)
        label_entries.append({
            "x": x, "y": y,
            "xy": (x * pie_radius * 0.65, y * pie_radius * 0.65),
            "label": f"{abbr(region)}: {share * 100:.0f}%",
        })

    # Keep all callouts in the reserved right-hand space of each pie panel.
    # This prevents a left-side label from spilling into the preceding panel.
    for side in (1,):
        entries = list(label_entries)
        entries.sort(key=lambda e: e["y"])
        if not entries:
            continue
        target_y = np.array([e["y"] * pie_radius * 1.12 for e in entries], dtype=float)
        min_sep = pie_radius * 0.34
        lower = -pie_radius * 1.05
        upper = pie_radius * 1.05
        target_y = np.clip(target_y, lower, upper)
        for j in range(1, len(target_y)):
            target_y[j] = max(target_y[j], target_y[j - 1] + min_sep)
        if target_y[-1] > upper:
            target_y -= target_y[-1] - upper
        for j in range(len(target_y) - 2, -1, -1):
            target_y[j] = min(target_y[j], target_y[j + 1] - min_sep)
        if target_y[0] < lower:
            target_y += lower - target_y[0]
        text_x = side * pie_radius * 1.32
        ha = "left" if side > 0 else "right"
        for entry, text_y in zip(entries, target_y):
            ax.annotate(
                    entry["label"], xy=entry["xy"], xytext=(text_x, text_y),
                    ha=ha, va="center", fontsize=6.3, fontweight="normal", color=COLORS["text"],
                    arrowprops=dict(arrowstyle="-", color=COLORS["spine"], linewidth=0.5,
                                    connectionstyle="arc3,rad=0"), annotation_clip=False)

    total_artist = ax.text(
        0, -pie_radius * 1.30, f"Total: {total_txt}",
        ha="center", va="top", fontsize=6.3, color=COLORS["subtle_text"],
    )
    total_artist.set_gid("figure5_pie_total")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(-pie_radius * 1.50, pie_radius * 1.35)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(-pie_radius * 0.1, pie_radius * 2.45)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================
# Country name helpers
# ============================================================
def _norm_name(s):
    if pd.isna(s):
        return ""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    for ch in [".", ",", "'", "`", "\u2019", "(", ")", "-"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())


MANUAL_COUNTRY_MAP = {
    "united states": "United States of America", "united states of america": "United States of America",
    "usa": "United States of America", "us": "United States of America",
    "russia": "Russia", "russian federation": "Russia",
    "south korea": "South Korea", "korea republic of": "South Korea",
    "republic of korea": "South Korea", "korea rep": "South Korea", "korea south": "South Korea",
    "north korea": "North Korea", "korea dem peoples rep": "North Korea", "korea dem rep": "North Korea",
    "democratic peoples republic of korea": "North Korea",
    "czech republic": "Czechia", "czechia": "Czechia", "slovak republic": "Slovakia",
    "kyrgyz republic": "Kyrgyzstan", "iran islamic rep": "Iran", "iran islamic republic of": "Iran",
    "egypt arab rep": "Egypt", "yemen rep": "Yemen", "syrian arab republic": "Syria",
    "lao pdr": "Laos", "lao peoples democratic republic": "Laos", "viet nam": "Vietnam",
    "brunei darussalam": "Brunei", "macedonia fyr": "North Macedonia", "macedonia": "North Macedonia",
    "north macedonia": "North Macedonia", "moldova": "Moldova", "republic of moldova": "Moldova",
    "cote d ivoire": "Ivory Coast", "ivory coast": "Ivory Coast",
    "congo dem rep": "Democratic Republic of the Congo",
    "democratic republic of the congo": "Democratic Republic of the Congo",
    "congo rep": "Republic of the Congo", "republic of the congo": "Republic of the Congo",
    "congo": "Republic of the Congo", "tanzania": "United Republic of Tanzania",
    "united republic of tanzania": "United Republic of Tanzania",
    "gambia the": "Gambia", "the gambia": "Gambia", "bahamas the": "The Bahamas", "bahamas": "The Bahamas",
    "hong kong sar china": "Hong Kong S.A.R.", "hong kong": "Hong Kong S.A.R.",
    "macao sar china": "Macao S.A.R", "macau": "Macao S.A.R", "macao": "Macao S.A.R",
    "turkey": "Turkey", "turkiye": "Turkey", "east timor": "Timor-Leste", "timor leste": "Timor-Leste",
    "cabo verde": "Cape Verde", "cape verde": "Cape Verde", "swaziland": "eSwatini", "eswatini": "eSwatini",
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "great britain": "United Kingdom",
    "venezuela rb": "Venezuela", "venezuela bolivarian republic of": "Venezuela",
    "bolivia plurinational state of": "Bolivia", "tanzania united republic of": "United Republic of Tanzania",
    "st lucia": "Saint Lucia", "st vincent and the grenadines": "Saint Vincent and the Grenadines",
    "st kitts and nevis": "Saint Kitts and Nevis", "micronesia fed sts": "Federated States of Micronesia",
    "micronesia federated states of": "Federated States of Micronesia",
    "serbia": "Republic of Serbia", "republic of serbia": "Republic of Serbia",
    "palestine": "Palestine", "west bank and gaza": "Palestine", "state of palestine": "Palestine",
    "vatican": "Vatican", "holy see": "Vatican", "bosnia and herzegovina": "Bosnia and Herzegovina",
    "myanmar": "Myanmar", "burma": "Myanmar", "dominican rep": "Dominican Republic", "slovakia": "Slovakia",
}


def build_world_name_lookup(world_gdf):
    name_cols = [c for c in ["NAME", "NAME_LONG", "ADMIN", "SOVEREIGNT", "NAME_EN",
                             "FORMAL_EN", "BRK_NAME", "NAME_CIAWF"] if c in world_gdf.columns]
    lookup = {}
    for _, row in world_gdf.iterrows():
        iso = row.get("ISO_A3")
        if pd.isna(iso) or iso in (None, "-99", ""):
            iso = row.get("ADM0_A3", None)
        for col in name_cols:
            v = row.get(col)
            if pd.notna(v):
                key = _norm_name(v)
                if key and key not in lookup:
                    lookup[key] = iso
    return lookup


def match_ww_countries_to_world(ww_countries, world_gdf):
    lookup = build_world_name_lookup(world_gdf)
    matched_iso = set(); unmatched = []; mapping = {}
    for c in ww_countries:
        if pd.isna(c):
            continue
        key = _norm_name(c); iso = None
        if key in MANUAL_COUNTRY_MAP:
            iso = lookup.get(_norm_name(MANUAL_COUNTRY_MAP[key]))
        if iso is None and key in lookup:
            iso = lookup[key]
        if iso is None:
            for lk_name, lk_iso in lookup.items():
                if key and (key in lk_name or lk_name in key) and len(key) >= 4:
                    iso = lk_iso; break
        if iso is not None and pd.notna(iso):
            matched_iso.add(iso); mapping[c] = iso
        else:
            unmatched.append(c)
    return matched_iso, unmatched, mapping


# ============================================================
# Load inputs
# ============================================================
global_df = pd.read_csv(input_code2_csv, encoding="utf-8-sig")
new_summary_columns = {
    "LAT_WWTP_deg",
    "LON_WWTP_deg",
    "sum_Qb_m3_day",
    "sum_Lb_m",
    "sum_Lb_over_Qb_day_per_m2",
}
if new_summary_columns.issubset(global_df.columns):
    global_df["LAT_WWTP"] = global_df["LAT_WWTP_deg"]
    global_df["LON_WWTP"] = global_df["LON_WWTP_deg"]
    global_df["sum_Q_m3_per_d"] = pd.to_numeric(
        global_df["sum_Qb_m3_day"], errors="coerce"
    )
    global_df["sum_L_m"] = global_df["sum_Lb_m"]
    global_df["sum_L_over_Q"] = pd.to_numeric(
        global_df["sum_Lb_over_Qb_day_per_m2"], errors="coerce"
    )
    print(
        "Mapped global_sewershed catchment_summary.csv to Figure 5 "
        "columns (Lb in m, Qb in m3/d and Lb/Qb in m day m-3)."
    )
if "ratio" not in global_df.columns:
    raise ValueError(
        "Column missing in catchment_summary.csv: ratio"
    )
global_df["ratio"] = pd.to_numeric(global_df["ratio"], errors="coerce")
ratio_keep = (
    np.isfinite(global_df["ratio"])
    & global_df["ratio"].between(
        MIN_SERVICE_POPULATION_RATIO,
        MAX_SERVICE_POPULATION_RATIO,
        inclusive="both",
    )
)
print(
    "[Service population ratio filter] "
    f"keeping {int(ratio_keep.sum()):,}/{len(global_df):,} catchments "
    f"with {MIN_SERVICE_POPULATION_RATIO} <= ratio <= "
    f"{MAX_SERVICE_POPULATION_RATIO}; excluded "
    f"{int((~ratio_keep).sum()):,}."
)
global_df = global_df.loc[ratio_keep].copy()
reg_df = pd.read_csv(input_reg_csv, encoding="utf-8-sig")
print("Loaded global catchment data:", global_df.shape)
print("Loaded regression equations:", reg_df.shape)

expected_regression_rows = reg_df[reg_df["y_var"].isin([
    "Problem_Length_km", "Toxic_Length_km", "Contaminated_Area_km2",
])]
if len(expected_regression_rows) != 3:
    raise ValueError("Figure 5 requires all three Figure 4 regression equations.")
if not (expected_regression_rows["x_var"] == "sum_L_build_over_Q").all():
    raise ValueError("Figure 5 regressions must use sum_L_build_over_Q.")
if "x_unit" not in expected_regression_rows.columns or not (
    expected_regression_rows["x_unit"].astype(str).str.strip() == "m day m-3"
).all():
    raise ValueError(
        "Figure 5 requires Figure 4 regressions with x_unit='m day m-3'; "
        "the supplied regression file is stale or uses annual Qb units."
    )

threshold_col = "corrosion_rate_threshold_mm_per_year"
if threshold_col not in reg_df.columns:
    raise ValueError(
        "Figure 5 requires a Figure 4 regression generated with the direct "
        "corrosion-rate criterion; threshold metadata is missing."
    )
problem_threshold = pd.to_numeric(
    reg_df.loc[reg_df["y_var"] == "Problem_Length_km", threshold_col],
    errors="coerce",
)
if problem_threshold.empty or not np.allclose(
    problem_threshold.to_numpy(dtype=float),
    CORROSION_RATE_THRESHOLD_MM_PER_YEAR,
    rtol=0.0,
    atol=1e-12,
    equal_nan=False,
):
    raise ValueError(
        "Figure 5 expected corrosion_rate_threshold_mm_per_year=1.0 in "
        "the Figure 4 regression."
    )

global_df["original_row_id"] = np.arange(len(global_df))

required_cols = ["LAT_WWTP", "LON_WWTP", "sum_Q_m3_per_d", "sum_L_over_Q"]
for c in required_cols:
    if c not in global_df.columns:
        raise ValueError(f"Column missing in code2 output: {c}")

global_df["LAT_WWTP"] = pd.to_numeric(global_df["LAT_WWTP"], errors="coerce")
global_df["LON_WWTP"] = pd.to_numeric(global_df["LON_WWTP"], errors="coerce")
global_df["sum_Q_m3_per_d"] = pd.to_numeric(global_df["sum_Q_m3_per_d"], errors="coerce")
global_df["sum_L_over_Q"] = pd.to_numeric(global_df["sum_L_over_Q"], errors="coerce")

sum_L_m_col = detect_sum_L_m_column(global_df)
if sum_L_m_col is None:
    print("[WARN] Could not detect sum_L_m column.")
    global_df["sum_L_m"] = np.nan
else:
    print(f"[Info] Detected sum_L_m column: {sum_L_m_col}")
    global_df["sum_L_m"] = pd.to_numeric(global_df[sum_L_m_col], errors="coerce")

# Outlier handling
sum_L_b_col = detect_sum_L_b_column(global_df)
global_df["filter_keep_sum_L_b"] = True
global_df["filter_keep_L_over_Q"] = True
global_df["filter_keep_all"] = True
global_df["filter_reason"] = ""

if sum_L_b_col is not None:
    global_df[sum_L_b_col] = pd.to_numeric(global_df[sum_L_b_col], errors="coerce")
    if FILTER_SUM_L_B_OUTLIERS:
        keep_sum_L_b, lb_lower, lb_upper = build_positive_quantile_filter(
            global_df[sum_L_b_col], qlow=SUM_L_B_QUANTILE_LOW,
            qhigh=SUM_L_B_QUANTILE_HIGH, abs_max=SUM_L_B_ABS_MAX)
        global_df["filter_keep_sum_L_b"] = keep_sum_L_b
        global_df.loc[~keep_sum_L_b, "filter_reason"] += f"abnormal_{sum_L_b_col}; "

global_df["sum_L_over_Q"] = pd.to_numeric(global_df["sum_L_over_Q"], errors="coerce")
global_df.loc[~np.isfinite(global_df["sum_L_over_Q"]), "sum_L_over_Q"] = np.nan
global_df.loc[global_df["sum_L_over_Q"] <= 0, "sum_L_over_Q"] = np.nan

if FILTER_L_OVER_Q_OUTLIERS:
    keep_lq, lq_lower, lq_upper = build_positive_quantile_filter(
        global_df["sum_L_over_Q"], qlow=L_OVER_Q_QUANTILE_LOW,
        qhigh=L_OVER_Q_QUANTILE_HIGH, abs_max=L_OVER_Q_ABS_MAX)
    global_df["filter_keep_L_over_Q"] = keep_lq
    global_df.loc[~keep_lq, "filter_reason"] += "abnormal_sum_L_over_Q; "

global_df["filter_keep_all"] = global_df["filter_keep_sum_L_b"] & global_df["filter_keep_L_over_Q"]

filtered_out_df = global_df[~global_df["filter_keep_all"]].copy()
if len(filtered_out_df) > 0:
    filtered_out_df.to_csv(os.path.join(output_dir, f"filtered_out_abnormal_Lb_LoverQ_{TAG}.csv"),
                           index=False, encoding="utf-8-sig")

if DROP_FILTERED_ROWS:
    global_df = global_df[global_df["filter_keep_all"]].copy()
elif REMOVE_OUTLIERS:
    bad_mask = ~global_df["filter_keep_all"]
    global_df.loc[bad_mask, ["sum_L_over_Q"]] = np.nan

# Regression
slope_prob, int_prob = get_regression_params(reg_df, "Problem_Length_km")
slope_tox, int_tox = get_regression_params(reg_df, "Toxic_Length_km")
slope_area, int_area = get_regression_params(reg_df, "Contaminated_Area_km2")

x_pred = global_df["sum_L_over_Q"].values
global_df["Problem_Length_km_pred"] = powerlaw_predict(x_pred, slope_prob, int_prob)
global_df["Toxic_Length_km_pred"] = powerlaw_predict(x_pred, slope_tox, int_tox)
global_df["Contaminated_Area_km2_pred"] = powerlaw_predict(x_pred, slope_area, int_area)
global_df["Contaminated_Area_m2_pred"] = global_df["Contaminated_Area_km2_pred"] * KM2_TO_M2
global_df["Corrosion_Rate_Threshold_mm_per_year"] = (
    CORROSION_RATE_THRESHOLD_MM_PER_YEAR
)

for col in ["Problem_Length_km_pred", "Toxic_Length_km_pred",
            "Contaminated_Area_m2_pred", "Contaminated_Area_km2_pred"]:
    global_df.loc[~np.isfinite(global_df[col]), col] = np.nan
    global_df.loc[global_df[col] <= 0, col] = np.nan

global_df["Toxic_to_Problem_ratio"] = global_df["Toxic_Length_km_pred"] / global_df["Problem_Length_km_pred"]
global_df.loc[~np.isfinite(global_df["Toxic_to_Problem_ratio"]), "Toxic_to_Problem_ratio"] = np.nan
global_df.loc[global_df["Toxic_to_Problem_ratio"] <= 0, "Toxic_to_Problem_ratio"] = np.nan

# Load world map
world = gpd.read_file(COUNTRIES_SHP).to_crs(epsg=4326)
world = world[world.geometry.notna()].copy()
world["geometry"] = world.geometry.buffer(0)
world["POP_EST"] = pd.to_numeric(world["POP_EST"], errors="coerce")
world["GDP_MD"] = pd.to_numeric(world["GDP_MD"], errors="coerce")

# Use every country represented by the map. Figure 5 no longer applies the
# former household-wastewater-collection threshold.
matched_iso = set(world["ISO_A3"].dropna())
excluded_world = world.iloc[0:0].copy()
kept_world = world.copy()
excluded_world["excl_reason"] = pd.Series(dtype=object)

# Spatial join
valid_coords = global_df.dropna(subset=["LON_WWTP", "LAT_WWTP"]).copy()
gdf_wwtp_all = gpd.GeoDataFrame(valid_coords,
    geometry=gpd.points_from_xy(valid_coords["LON_WWTP"], valid_coords["LAT_WWTP"]),
    crs="EPSG:4326")
world_info = world[["geometry", "NAME", "ISO_A3", "POP_EST", "GDP_MD", "SUBREGION"]].copy()
wwtp_country_all = gpd.sjoin(gdf_wwtp_all, world_info, how="left", predicate="within")
wwtp_country_all = wwtp_country_all.dropna(subset=["ISO_A3"]).copy()
wwtp_country = wwtp_country_all[wwtp_country_all["ISO_A3"].isin(matched_iso)].copy()

kept_indices = wwtp_country.index
global_df = global_df.loc[global_df.index.intersection(kept_indices)].copy()
sub_unique = wwtp_country[~wwtp_country.index.duplicated(keep="first")]
global_df["SUBREGION"] = global_df.index.map(sub_unique["SUBREGION"])
global_df["ISO_A3"] = global_df.index.map(sub_unique["ISO_A3"])

subregion_counts = global_df["SUBREGION"].value_counts()
retained_subregions = subregion_counts[
    subregion_counts >= MIN_SUBREGION_CATCHMENTS
].index
removed_subregions = subregion_counts[
    subregion_counts < MIN_SUBREGION_CATCHMENTS
].sort_values()
print(
    "[Subregion sample-size filter] "
    f"retaining {len(retained_subregions)} subregions with "
    f"n >= {MIN_SUBREGION_CATCHMENTS}; removing "
    f"{len(removed_subregions)} subregions and "
    f"{int(removed_subregions.sum()):,} catchments."
)
if len(removed_subregions):
    print(
        "[Subregion sample-size filter] removed: "
        + ", ".join(
            f"{region} (n={int(count)})"
            for region, count in removed_subregions.items()
        )
    )
global_df = global_df[
    global_df["SUBREGION"].isin(retained_subregions)
].copy()
wwtp_country = wwtp_country.loc[
    wwtp_country.index.intersection(global_df.index)
].copy()

# ============================================================
# Region colors
# ============================================================
country_subset = wwtp_country.dropna(subset=["SUBREGION"])
all_regions = sorted(country_subset["SUBREGION"].dropna().unique())
region_colors = {}
for r in all_regions:
    region_colors[r] = REGION_PALETTE[len(region_colors) % len(REGION_PALETTE)]
for r in sorted(world["SUBREGION"].dropna().unique()):
    if r not in region_colors:
        region_colors[r] = REGION_PALETTE[len(region_colors) % len(REGION_PALETTE)]

# ============================================================
# Aggregate by SUBREGION
# ============================================================
catchment_subregion = wwtp_country[[
    "SUBREGION", "Problem_Length_km_pred", "Toxic_Length_km_pred",
    "Contaminated_Area_m2_pred", "Contaminated_Area_km2_pred"]].copy()
catchment_subregion = catchment_subregion.dropna(subset=["SUBREGION"]).copy()

SUBREGION_agg = catchment_subregion.groupby("SUBREGION").agg(
    total_problem=("Problem_Length_km_pred", lambda x: x.sum(min_count=1)),
    total_toxic=("Toxic_Length_km_pred", lambda x: x.sum(min_count=1)),
    total_area=("Contaminated_Area_m2_pred", lambda x: x.sum(min_count=1)),
    total_area_km2=("Contaminated_Area_km2_pred", lambda x: x.sum(min_count=1)),
    problem_per_catchment=("Problem_Length_km_pred", "mean"),
    toxic_per_catchment=("Toxic_Length_km_pred", "mean"),
    area_per_catchment=("Contaminated_Area_m2_pred", "mean"),
    problem_per_catchment_sd=("Problem_Length_km_pred", "std"),
    toxic_per_catchment_sd=("Toxic_Length_km_pred", "std"),
    area_per_catchment_sd=("Contaminated_Area_m2_pred", "std"),
    problem_per_catchment_n=("Problem_Length_km_pred", "count"),
    toxic_per_catchment_n=("Toxic_Length_km_pred", "count"),
    area_per_catchment_n=("Contaminated_Area_m2_pred", "count"),
    total_catchment=("SUBREGION", "size"),
).reset_index()

SUBREGION_agg["toxic_problem_ratio"] = SUBREGION_agg["total_toxic"] / SUBREGION_agg["total_problem"]
SUBREGION_agg["Corrosion_Rate_Threshold_mm_per_year"] = (
    CORROSION_RATE_THRESHOLD_MM_PER_YEAR
)
SUBREGION_agg["problem_per_catchment_sem"] = (SUBREGION_agg["problem_per_catchment_sd"]
    / np.sqrt(SUBREGION_agg["problem_per_catchment_n"].replace(0, np.nan)))
SUBREGION_agg["toxic_per_catchment_sem"] = (SUBREGION_agg["toxic_per_catchment_sd"]
    / np.sqrt(SUBREGION_agg["toxic_per_catchment_n"].replace(0, np.nan)))
SUBREGION_agg["area_per_catchment_sem"] = (SUBREGION_agg["area_per_catchment_sd"]
    / np.sqrt(SUBREGION_agg["area_per_catchment_n"].replace(0, np.nan)))
SUBREGION_agg["area_per_catchment_km2"] = SUBREGION_agg["area_per_catchment"] / KM2_TO_M2
SUBREGION_agg["area_per_catchment_km2_sem"] = SUBREGION_agg["area_per_catchment_sem"] / KM2_TO_M2

for col in ["problem_per_catchment", "toxic_per_catchment", "area_per_catchment",
            "toxic_problem_ratio", "problem_per_catchment_sem",
            "toxic_per_catchment_sem", "area_per_catchment_sem"]:
    SUBREGION_agg.loc[~np.isfinite(SUBREGION_agg[col]), col] = np.nan
for col in ["problem_per_catchment", "toxic_per_catchment", "area_per_catchment", "toxic_problem_ratio"]:
    SUBREGION_agg.loc[SUBREGION_agg[col] <= 0, col] = np.nan
for col in ["problem_per_catchment_sem", "toxic_per_catchment_sem", "area_per_catchment_sem"]:
    SUBREGION_agg.loc[SUBREGION_agg[col] < 0, col] = 0.0
    SUBREGION_agg[col] = SUBREGION_agg[col].fillna(0.0)

region_order_d = (SUBREGION_agg.dropna(subset=["problem_per_catchment"])
                  .sort_values("problem_per_catchment", ascending=True)["SUBREGION"].tolist())

# Shared color scales
LQ_vmin, LQ_vmax = robust_positive_limits(global_df["sum_L_over_Q"].values, 2, 98)


def _make_cbar(ax_map, width="2.8%", height="88%", loc="center right", bbox=(0.06, 0., 1, 1)):
    return inset_axes(ax_map, width=width, height=height, loc=loc,
                      bbox_to_anchor=bbox, bbox_transform=ax_map.transAxes, borderpad=0)


def align_fig5_axes(fig, ax_map_a, cax_a, bar_axes, pie_axes, relationship_axes):
    fig.canvas.draw()
    pos_map = ax_map_a.get_position(); pos_cbar = cax_a.get_position()
    left_target = pos_map.x0; right_target = pos_cbar.x1
    pos_b = bar_axes[0].get_position(); pos_c = bar_axes[1].get_position()
    pos_d = bar_axes[2].get_position(); pos_h = relationship_axes[0].get_position()
    gap_bc = pos_c.x0 - pos_b.x1; gap_cd = pos_d.x0 - pos_c.x1; gap_dh = pos_h.x0 - pos_d.x1
    total_gap = gap_bc + gap_cd + gap_dh
    original_widths = np.array([pos_b.width, pos_c.width, pos_d.width, pos_h.width], dtype=float)
    ratios = original_widths / original_widths.sum()
    total_width = right_target - left_target - total_gap
    widths = total_width * ratios
    x0_b = left_target
    x0_c = x0_b + widths[0] + gap_bc
    x0_d = x0_c + widths[1] + gap_cd
    x0_h = x0_d + widths[2] + gap_dh
    for ax, x0, w in zip(bar_axes, [x0_b, x0_c, x0_d], widths[:3]):
        p = ax.get_position(); ax.set_position([x0, p.y0, w, p.height])
    for ax, x0, w in zip(pie_axes, [x0_b, x0_c, x0_d], widths[:3]):
        p = ax.get_position()
        ax.set_position([x0, p.y0 - 0.035, w, p.height])
    for rel_index, ax_rel in enumerate(relationship_axes):
        p_rel = ax_rel.get_position()
        rel_height = p_rel.height * 0.92
        rel_y0 = p_rel.y1 - rel_height if rel_index == 0 else p_rel.y0
        ax_rel.set_position(
            [x0_h, rel_y0, right_target - x0_h, rel_height]
        )
    for ax in bar_axes:
        ax.title.set_y(FIG1_BAR_TITLE_Y); ax.title.set_ha("left")
        ax.title.set_position((0.0, FIG1_BAR_TITLE_Y))
    for ax in pie_axes:
        ax.set_aspect("equal", adjustable="datalim")
        ax.title.set_y(FIG1_PIE_TITLE_Y); ax.title.set_ha("left")
        ax.title.set_position((0.0, FIG1_PIE_TITLE_Y))
    fig.canvas.draw()


# ============================================================
# Figure 5
# ============================================================
FIG5_SIZE = (16.8 / 1.3, 13.2 / 1.3)
FIG5_HEIGHT_R = [1.12, 0.78, 0.48]
FIG5_WIDTH_R = [1.0, 1.0, 1.0, 2.75]
FIG5_HSPACE = 0.32
FIG5_WSPACE = 0.55
FIG5_MARGINS = dict(left=0.055, right=0.955, top=0.965, bottom=0.170)

fig = plt.figure(figsize=FIG5_SIZE)
gs = GridSpec(3, 4, figure=fig, width_ratios=FIG5_WIDTH_R, height_ratios=FIG5_HEIGHT_R,
              hspace=FIG5_HSPACE, wspace=FIG5_WSPACE)

ax_map_a = fig.add_subplot(gs[0, :])
ax_bar_b = fig.add_subplot(gs[1, 0])
ax_bar_c = fig.add_subplot(gs[1, 1])
ax_bar_d = fig.add_subplot(gs[1, 2])
ax_pie_e = fig.add_subplot(gs[2, 0])
ax_pie_f = fig.add_subplot(gs[2, 1])
ax_pie_g = fig.add_subplot(gs[2, 2])
relationship_gs = gs[1:, 3].subgridspec(2, 1, hspace=0.32)
ax_rel_h = fig.add_subplot(relationship_gs[0, 0])
ax_rel_i = fig.add_subplot(relationship_gs[1, 0])

cax_a = _make_cbar(ax_map_a, width="1.2%", height="80%", bbox=(0.04, 0., 1, 1))

# --- panel a ---
plot_map_panel(ax_map_a, world, global_df, "sum_L_over_Q", "a",
               r"Conveyance burden index, $\sum L_b/Q_b$", cax=cax_a,
               cbar_label=r"$\sum L_b/Q_b$ (m d m$^{-3}$)", cmap=CMAP_LQ,
               vmin=LQ_vmin, vmax=LQ_vmax, excluded_world=excluded_world,
               show_excluded_legend=False, **get_panel_size_config("a"))

# --- panel b ---
plot_region_bar_ordered_with_error(ax_bar_b, SUBREGION_agg, y_col="problem_per_catchment",
    err_col="problem_per_catchment_sem", total_col_for_pie="total_problem",
    n_col="problem_per_catchment_n",
    xlabel="Corrosive pipe length\nper catchment (km)",
    panel_label="b", panel_title="Corrosion per catchment",
    region_colors=region_colors,
    region_order=region_order_d, panel_cmap=CMAP_PANEL_D, xlim_max=None, show_n=True)

# --- panel c ---
plot_region_bar_ordered_with_error(ax_bar_c, SUBREGION_agg, y_col="toxic_per_catchment",
    err_col="toxic_per_catchment_sem", total_col_for_pie="total_toxic",
    n_col="toxic_per_catchment_n", xlabel="Toxic length\nper catchment (km)",
    panel_label="c", panel_title="Toxic per catchment", region_colors=region_colors,
    region_order=None, panel_cmap=CMAP_PANEL_E, xlim_max=None, show_n=True)

# --- panel d ---
plot_region_bar_ordered_with_error(ax_bar_d, SUBREGION_agg, y_col="area_per_catchment_km2",
    err_col="area_per_catchment_km2_sem", total_col_for_pie="total_area_km2",
    n_col="area_per_catchment_n", xlabel="Odour affected area\nper catchment (km²)",
    panel_label="d", panel_title="Odour per catchment", region_colors=region_colors,
    region_order=None, panel_cmap=CMAP_PANEL_F, show_n=True)

# --- panel e ---
plot_region_share_pie_panel(ax_pie_e, SUBREGION_agg, y_col="problem_per_catchment",
    err_col="problem_per_catchment_sem", total_col_for_pie="total_problem",
    n_col="problem_per_catchment_n", panel_label="e",
    panel_title="Corrosion total share",
    region_colors=region_colors, region_order=region_order_d, panel_cmap=CMAP_PANEL_D,
    pie_radius=FIG1_PIE_RADIUS, extra_label_regions=FIG1_PIE_LABEL_REGIONS)

# --- panel f ---
plot_region_share_pie_panel(ax_pie_f, SUBREGION_agg, y_col="toxic_per_catchment",
    err_col="toxic_per_catchment_sem", total_col_for_pie="total_toxic",
    n_col="toxic_per_catchment_n", panel_label="f", panel_title="Toxic total share",
    region_colors=region_colors, region_order=None, panel_cmap=CMAP_PANEL_E,
    pie_radius=FIG1_PIE_RADIUS, extra_label_regions=FIG1_PIE_LABEL_REGIONS)

# --- panel g ---
plot_region_share_pie_panel(ax_pie_g, SUBREGION_agg, y_col="area_per_catchment",
    err_col="area_per_catchment_sem", total_col_for_pie="total_area_km2",
    n_col="area_per_catchment_n", panel_label="g", panel_title="Odour total share",
    region_colors=region_colors, region_order=None, panel_cmap=CMAP_PANEL_F,
    pie_radius=FIG1_PIE_RADIUS, extra_label_regions=FIG1_PIE_LABEL_REGIONS)

# --- panel h ---
PANEL_H_COLORS = {
    "Eastern Europe": "#255C72", "Western Europe": "#2D708B", "Northern Europe": "#388CAE",
    "Southern Europe": "#7CA5B8", "Northern America": "#6AAC9C", "Eastern Asia": "#472F2D",
    "Central America": "#A89374", "South America": "#C2B39E", "Northern Africa": "#A1B4AC",
    "Western Asia": "#AA6526", "Southern Africa": "#5C746A", "South-Eastern Asia": "#96433A",
    "Australia and New Zealand": "#82BFB0",
}

plot_relationship_scatter(
    ax_rel_h, global_df,
    x_col="average_Qb_m3_day",
    y_col="sum_L_m",
    region_col="SUBREGION", panel_label="h",
    panel_title=r"$\sum L_b$ vs. average $Q_b$",
    region_colors=PANEL_H_COLORS, use_log=FIGURE_H_LOG_SCALE, error_type="sem",
    center_type="mean",
    linear_x_scale=1.0, linear_y_scale=1e9,
    x_label_log=r"Average $Q_b$ (m$^3$ d$^{-1}$)",
    y_label_log=r"$\sum L_b$ (m)",
    x_label_linear=r"Average $Q_b$ (m$^3$ d$^{-1}$)",
    y_label_linear=r"$\sum L_b$ ($10^9$ m)",
    size_col="sum_L_over_Q",
    size_label=r"$\sum L_b/Q_b$",
    show_region_legend=False,
    size_legend_loc="upper left",
    show_size_legend=True,
    annotate_region_sizes=False,
)

# --- panel i ---
plot_relationship_scatter(
    ax_rel_i, global_df,
    x_col="per_capita_l_day",
    y_col="average_assigned_population_per_building",
    region_col="SUBREGION", panel_label="i",
    panel_title="Population vs. per-capita flow",
    region_colors=PANEL_H_COLORS, use_log=FIGURE_H_LOG_SCALE, error_type="sem",
    center_type="mean",
    linear_x_scale=1.0, linear_y_scale=1.0,
    x_label_log=r"Per-capita wastewater (L person$^{-1}$ d$^{-1}$)",
    y_label_log="Average assigned population per building",
    x_label_linear=r"Per-capita wastewater (L person$^{-1}$ d$^{-1}$)",
    y_label_linear="Average assigned population per building",
    size_col="average_Qb_m3_day",
    size_label=r"Average $Q_b$",
    region_legend_loc="upper center",
    region_legend_bbox=(0.5, -0.24),
    region_legend_ncol=4,
    region_legend_frame=False,
    region_legend_title=None,
    include_size_in_region_legend=False,
    show_region_legend=True,
    size_legend_loc="upper left",
    show_size_legend=True,
    annotate_region_sizes=False,
)

ax_rel_h.set_ylim(1e5, 1e12)
ax_rel_i.set_ylim(1e-1, 1e3)

fig.subplots_adjust(**FIG5_MARGINS)
align_fig5_axes(fig, ax_map_a, cax_a, [ax_bar_b, ax_bar_c, ax_bar_d],
                [ax_pie_e, ax_pie_f, ax_pie_g], [ax_rel_h, ax_rel_i])

_h_axes_tag = "LOG" if FIGURE_H_LOG_SCALE else "LIN"
_size_tag = SCATTER_SIZE_MODE.upper()

fig_base = os.path.join(
    output_dir,
    f"FIG5_main_9panels_{TAG}_OUTLIER_{_suffix}_hi{_h_axes_tag}_S{_size_tag}",
)
save_extra_artists = [
    artist for artist in ax_rel_i.get_children()
    if artist.get_gid() == "figure5_region_legend"
]
fig.savefig(
    fig_base + ".png", dpi=600, bbox_inches="tight",
    bbox_extra_artists=save_extra_artists,
)
if NO_SHOW:
    plt.close(fig)
else:
    plt.show()

# ============================================================
# Export data for the Results-text script
# ============================================================
catchment_export_cols = [
    "ISO_A3", "SUBREGION", "LON_WWTP", "LAT_WWTP",
    "ratio", "average_assigned_population_per_building",
    "average_Qb_m3_day", "per_capita_l_day",
    "sum_L_over_Q", "sum_L_m", "sum_Q_m3_per_d",
    "Problem_Length_km_pred", "Toxic_Length_km_pred",
    "Contaminated_Area_m2_pred", "Contaminated_Area_km2_pred",
    "Toxic_to_Problem_ratio",
    "Corrosion_Rate_Threshold_mm_per_year",
]
catchment_export_cols = [c for c in catchment_export_cols if c in global_df.columns]
fig5_catchment_export = global_df[catchment_export_cols].copy()

fig5_catchment_csv = os.path.join(
    output_dir, f"figure5_catchment_data_{TAG}_OUTLIER_{_suffix}.csv")
fig5_catchment_export.to_csv(fig5_catchment_csv, index=False, encoding="utf-8-sig")

fig5_subregion_csv = os.path.join(
    output_dir, f"figure5_subregion_agg_{TAG}_OUTLIER_{_suffix}.csv")
SUBREGION_agg.to_csv(fig5_subregion_csv, index=False, encoding="utf-8-sig")

print("\n[Export] Data exported for the Results-text script:")
print(f"  - {fig5_catchment_csv}")
print(f"  - {fig5_subregion_csv}")
print(f"\nFigure 5 completed. (REMOVE_OUTLIERS = {REMOVE_OUTLIERS}, "
      f"FIGURE_H_LOG_SCALE = {FIGURE_H_LOG_SCALE})")
