# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:36:15 2026

@author: zouxu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Figure 2 and its audit tables from package-local inputs."""

import argparse
import os
import warnings
import math
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from shapely.geometry import Point, LineString
from pyproj import CRS

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# SWITCHES (quick toggles)
# ═══════════════════════════════════════════════════════════════════════════════
PANEL_B_USE_LOG = False       # True -> log-log axes for verification plot ; False -> linear
LAYOUT_PREVIEW_ONLY = False   # True -> draw only axis frames/layout (fast check)

# >>> 新增：figure d, g, j（Atmospheric dispersion 面板）的散点半径放大倍数 <<<
# 设为 10 表示画成真实半径的 10 倍，设为 20 表示 20 倍。设为 1.0 即真实大小。
DISP_RADIUS_SCALE = 50.0

# Panels d/g/j only: remove the largest dispersion-radius outliers before
# drawing circles. This value is a percent; set 0 to disable.
DISP_RADIUS_OUTLIER_PERCENT = 0#0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 0. GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════════════
BG_FIG = "white"
BG_AX = "white"
GRID_C = "#D9D4CB"
SPINE_C = "#333333"
TEXT_C = "#222222"
SUBTEXT_C = "#666666"

SAGE_LIGHT = "#DCE6E2"
SAGE = "#A1B4AC"
SAGE_DARK = "#82BFB0"
TEAL = "#3A7A8C"
NAVY = "#08345A"
BEIGE = "#E7E3DA"
BROWN = "#A66842"
ACCENT_RED = "#9C3106"
WARM_SOFT = "#D8CFC1"

SCI_WHISKER_BAND_COLOR = "#CDBFA8"
SCI_WHISKER_BAND_ALPHA = 0.18

CITY_COLORS = {
    "Hong Kong": TEAL,
    "Toronto": BROWN,
    "Los Angeles": SAGE
}

BORDER_BG_COLOR = BEIGE
BORDER_BG_ALPHA = 0.55


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — PACKAGE-LOCAL FILE PATHS
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
FIG2_INPUT_DIR = SCRIPT_DIR / "data" / "figure2"

OUTPUT_DIR = str(SCRIPT_DIR / "figure_2_1")

# NEW: where the exported data for the results-calculator script will go
FIG2_DATA_DIR = os.path.join(OUTPUT_DIR, "figure2_data")


def _path(*parts):
    return str(Path(*parts))


def _first_existing(*paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return str(Path(path))
    fallback = next((path for path in paths if path is not None), None)
    return str(Path(fallback)) if fallback is not None else None


DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / f"{SCRIPT_DIR.name}_results"
HK_BIO_DIR = DEFAULT_RESULTS_ROOT / "HK_v3" / "biochemical_results"
TOR_BIO_DIR = DEFAULT_RESULTS_ROOT / "toronto_v3" / "biochemical_results"
LA_BIO_DIR = DEFAULT_RESULTS_ROOT / "LA_v3" / "biochemical_results"

SCI_WHISKER_CSV = _first_existing(
    DEFAULT_RESULTS_ROOT / "figure1" / "hk_whisker_ranges.csv",
)

HK_CFG = dict(
    name="Hong Kong",
    rp_csv=_path(HK_BIO_DIR / "hk_result_segments_v7.csv"),
    c006_csv=_path(HK_BIO_DIR / "hk_emission_radius_segments_v7.csv"),
    c05_csv=_path(HK_BIO_DIR / "hk_emission_radius_odour_segments_v7.csv"),
    border_shp=_first_existing(
        os.environ.get("HRSNM_FIG2_HK_BORDER_SHP"),
        FIG2_INPUT_DIR / "HK_border" / "hk_merged_border.shp",
    ),
    border_crs="EPSG:2326",
    maint_shp=_first_existing(
        os.environ.get("HRSNM_FIG2_HK_MAINT_SHP"),
        FIG2_INPUT_DIR / "HK_maintenance" / "Maintenance_project.shp",
    ),
    source_crs="EPSG:2326",
    plot_crs="EPSG:2326",
    xlim=(807000, 850000),
    ylim=(807000, 847000),
    scalebar_m=2000,
)

TOR_CFG = dict(
    name="Toronto",
    rp_csv=_path(TOR_BIO_DIR / "toronto_result_segments_v7.csv"),
    c006_csv=_path(TOR_BIO_DIR / "toronto_emission_radius_segments_v7.csv"),
    c05_csv=_path(TOR_BIO_DIR / "toronto_emission_radius_odour_segments_v7.csv"),
    border_shp=_first_existing(
        os.environ.get("HRSNM_FIG2_TORONTO_BORDER_SHP"),
        FIG2_INPUT_DIR / "Toronto_border" / "citygcs_regional_mun_wgs84.shp",
    ),
    border_crs="EPSG:4326",
    maint_shp=None,
    source_crs="EPSG:26917",
    plot_crs="EPSG:26917",
    xlim=(607000, 652000),
    ylim=(4823000, 4860000),
    scalebar_m=None,
)

LA_CFG = dict(
    name="Los Angeles",
    rp_csv=_path(LA_BIO_DIR / "la_result_segments_v7.csv"),
    c006_csv=_path(LA_BIO_DIR / "la_emission_radius_segments_v7.csv"),
    c05_csv=_path(LA_BIO_DIR / "la_emission_radius_odour_segments_v7.csv"),
    border_shp=_first_existing(
        os.environ.get("HRSNM_FIG2_LA_BORDER_SHP"),
        FIG2_INPUT_DIR / "LA_border" / "City_Boundary.shp",
    ),
    border_crs="EPSG:4326",
    maint_shp=None,
    source_crs="EPSG:26945",
    plot_crs="EPSG:26945",
    xlim=(1935000, 1990000),
    ylim=(545000, 595000),
    scalebar_m=None,
)

TAG = "Figure2_merged_params_wwtp_maps"
DPI = 600


def parse_args():
    default_root = DEFAULT_RESULTS_ROOT
    parser = argparse.ArgumentParser(description="Generate HRSNM Figure 2.")
    parser.add_argument("--results-root", type=Path, default=default_root)
    parser.add_argument("--whisker-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=default_root / "figure2")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# CODE-2 CONFIG (WWTP predicted vs measured)
# ═══════════════════════════════════════════════════════════════════════════════
MEASURED_XLSX = os.environ.get(
    "HRSNM_FIG2_MEASURED_XLSX",
    str(FIG2_INPUT_DIR / "17WWTP_inflow_data1.xlsx"),
)

FLOW_UNIT_FACTOR = 86.4  # m3/s -> ML/d

HK_NODE_TO_CATCHMENT = {
    "Sha_Tin_STW":                  "Shatin STW",
    "Tai_Po_Sewage_Treatment_Works": "Taipo STW",
    "Shek_Wu_Hui_STW":              "Shek Wu Hui STW",
    "Yuen_Long_STW":                "Yuen Long STW",
    "San_Wai_STW":                  "San Wai STW",
    "Pillar_Point_STW":             "Pillar Point STW",
    "Sham_Tseng_STW":               "Sham Tseng STW",
    "SCISTW":                       "Stonecutter island STW",
    "Sai_Kung_STW":                 "Sai Kung STW",
}

TOR_NODE_TO_CATCHMENT = {
    "WWTP1": "Ashbridges Bay Wastewater Treatment Plant",
    "WWTP2": "North Toronto Wastewater Treatment Plant",
    "WWTP3": "Humber Wastewater Treatment Plant",
    "WWTP4": "Highland Creek Treatment Plant",
    "WWTP5": "Lakeview Wastewater Treatment Plant",
}

LA_NODE_TO_CATCHMENT = {
    "WWTP1": "Hyperion Water Reclamation Plant",
    "WWTP3": "Los Angeles-Glendale Reclamation Plant",
    "WWTP4": "Donald C Tillman Water Reclamation Plant",
}

CITY_NODE_MAPS = {
    "Hong Kong":   (HK_CFG["rp_csv"],  HK_NODE_TO_CATCHMENT),
    "Toronto":     (TOR_CFG["rp_csv"], TOR_NODE_TO_CATCHMENT),
    "Los Angeles": (LA_CFG["rp_csv"],  LA_NODE_TO_CATCHMENT),
}

MEASURED_FALLBACK = {
    "Shatin STW": 250.8,
    "Taipo STW": 104.4,
    "Shek Wu Hui STW": 80.4,
    "Yuen Long STW": 23.5,
    "San Wai STW": 136.1,
    "Pillar Point STW": 179.5,
    "Sham Tseng STW": 8.7,
    "Stonecutter island STW": 1916.490458,
    "Sai Kung STW": 8.5,
    "Ashbridges Bay Wastewater Treatment Plant": 576.29,
    "North Toronto Wastewater Treatment Plant": 18.3,
    "Humber Wastewater Treatment Plant": 286.5,
    "Highland Creek Treatment Plant": 184.8,
    "Lakeview Wastewater Treatment Plant": 35.0,
    "Hyperion Water Reclamation Plant": 1040.9,
    "Los Angeles-Glendale Reclamation Plant": 212.0,
    "Donald C Tillman Water Reclamation Plant": 394.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════════════
try:
    plt.rcParams["font.family"] = "DejaVu Sans"
except Exception:
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams.update({
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "axes.titleweight": "normal",
    "legend.fontsize": 5.8,
    "legend.frameon": True,
    "legend.edgecolor": "none",
    "legend.fancybox": False,
    "figure.facecolor": BG_FIG,
    "axes.facecolor": BG_AX,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "axes.unicode_minus": False,
})


# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
H2S_LABELS = ["0 – 10", "10 – 50", "50 – 100", "> 100"]
H2S_BINS = [0, 10, 50, 100, np.inf]
H2S_COLORS = [SAGE_LIGHT, SAGE, TEAL, NAVY]

H2S_MS_HK = [0.2, 0.4, 0.75, 1.25]
H2S_MS_DEF = [0.3, 0.3, 0.3, 0.3]

CLR_R006 = "#7A3F22"
CLR_R05 = "#6F1D04"

DISP_CLR_01 = CLR_R05
DISP_CLR_001 = CLR_R006

DISP_ALPHA_01 = 0.35
DISP_ALPHA_001 = 0.22

DISP_LEGEND_ALPHA_01 = 0.45
DISP_LEGEND_ALPHA_001 = 0.26

SCATTER_SCALE_006 = 0.10
SCATTER_SCALE_05 = 0.10

CORR_LABELS = ["0 – 0.5", "0.5 – 1", "> 1"]
CORR_BINS = [0, 0.5, 1, np.inf]
CORR_COLORS = [SAGE, BROWN, NAVY]
CORR_LWS = [0.45, 1.00, 1.05]

MAINT_CLR = NAVY
BG_PIPE_CLR = "#CFCAC2"

DISP_PIPE_CLR = "#E3E0DA"
DISP_PIPE_LW = 0.08
DISP_PIPE_ALPHA = 0.45

BORDER_CLR = SPINE_C
BORDER_LW = 0.6

WWTP_COLOR = "black"
WWTP_MARKER_SIZE = 10


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def apply_axis_style(ax, facecolor=BG_AX, half_frame=False):
    ax.set_facecolor(facecolor)

    if half_frame:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_color(BORDER_CLR)
        ax.spines["bottom"].set_color(BORDER_CLR)
        ax.spines["left"].set_linewidth(0.7)
        ax.spines["bottom"].set_linewidth(0.7)
    else:
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(BORDER_CLR)
            s.set_linewidth(0.7)

    ax.tick_params(axis="both", colors=TEXT_C, labelsize=6,
                   direction="in", width=0.5, top=False, right=False)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color(TEXT_C)


def _layout_placeholder(ax, label="", title=""):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(SPINE_C)
        s.set_linewidth(0.8)
    ax.set_facecolor("white")
    if title:
        ax.set_title(title, fontsize=7, color=SUBTEXT_C, pad=4)
    _add_panel_label(ax, label)


# ═══════════════════════════════════════════════════════════════════════════════
# SCI WHISKER RANGES
# ═══════════════════════════════════════════════════════════════════════════════
def load_sci_whisker_ranges(csv_path):
    if not os.path.exists(csv_path):
        print(f"⚠ SCI whisker ranges CSV not found at: {csv_path}")
        print("  本次将不绘制 SCI whisker band。")
        return {}

    df = pd.read_csv(csv_path)

    required = {"parameter", "whisker_min", "whisker_max"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"SCI whisker CSV missing required columns: {missing}. "
            f"Required columns are: {required}"
        )

    out = {}
    for _, row in df.iterrows():
        p = str(row["parameter"])
        wmin = float(row["whisker_min"])
        wmax = float(row["whisker_max"])
        out[p] = (wmin, wmax)

    print(f"✓ Loaded SCI whisker ranges from: {csv_path}")
    for k, v in out.items():
        print(f"    {k}: [{v[0]:.4g}, {v[1]:.4g}]")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CRS / PROJECTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def detect_lonlat_from_xy(x, y):
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    if len(x) == 0 or len(y) == 0:
        return False
    return (x.between(-180, 180).mean() > 0.98) and \
           (y.between(-90, 90).mean() > 0.98)


def utm_epsg_from_lonlat(lon, lat):
    zone = int(math.floor((lon + 180) / 6) + 1)
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def choose_local_projected_crs_from_xy(x, y, city_name=None):
    if not detect_lonlat_from_xy(x, y):
        return None
    lon = float(pd.Series(x).dropna().median())
    lat = float(pd.Series(y).dropna().median())
    epsg = utm_epsg_from_lonlat(lon, lat)
    print(f"  [{city_name}] detected lon/lat; chosen projected CRS = {epsg}")
    return epsg


def is_metric_crs(crs_like):
    if crs_like is None:
        return False
    try:
        crs = CRS.from_user_input(crs_like)
        for ax in crs.axis_info:
            if "metre" in ax.unit_name.lower() or "meter" in ax.unit_name.lower():
                return True
        return False
    except Exception:
        return False


def auto_scalebar_length_m(xlim):
    width = abs(xlim[1] - xlim[0])
    target = width * 0.18
    if target <= 0:
        return None
    nice_vals = np.array(
        [100, 200, 500, 1000, 2000, 5000,
         10000, 20000, 50000, 100000, 200000],
        dtype=float
    )
    return float(nice_vals[np.argmin(np.abs(nice_vals - target))])


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_csvs(cfg):
    rp = pd.read_csv(cfg["rp_csv"])
    c006 = pd.read_csv(cfg["c006_csv"])
    c05 = pd.read_csv(cfg["c05_csv"])

    safe_numeric(
        rp,
        ["us_x", "us_y", "ds_x", "ds_y", "v", "velocity", "flowrate",
         "WIDTH", "width", "diameter", "DIAMETER", "Diameter", "slope", "depth"]
    )
    safe_numeric(c006, ["us_x", "us_y", "distance", "SH2S", "dcorr_dt"])
    safe_numeric(c05, ["us_x", "us_y", "distance"])

    print(f"  [{cfg['name']}] rp={len(rp):,}  c006={len(c006):,}  c05={len(c05):,}")
    return rp, c006, c05


def load_shapefile(path, src_crs=None, dst_crs=None):
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"Shapefile not found, skipping optional layer: {path}")
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None and src_crs is not None:
        gdf = gdf.set_crs(src_crs)
    if dst_crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(dst_crs)
    return gdf


def load_border(cfg, dst_crs=None):
    path = cfg.get("border_shp")
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"⚠ Border shapefile not found for {cfg['name']}: {path}")
        return None
    try:
        gdf = load_shapefile(path, src_crs=cfg.get("border_crs"), dst_crs=dst_crs)
        print(f"  [{cfg['name']}] border loaded: {len(gdf)} feature(s), crs -> {gdf.crs}")
        return gdf
    except Exception as e:
        print(f"⚠ Failed to load border for {cfg['name']}: {e}")
        return None


def draw_border_basemap(ax, border_gdf, color=BORDER_BG_COLOR, alpha=BORDER_BG_ALPHA):
    if border_gdf is None or len(border_gdf) == 0:
        return
    try:
        border_gdf.plot(
            ax=ax,
            facecolor=color,
            edgecolor="none",
            linewidth=0,
            alpha=alpha,
            zorder=0,
        )
    except Exception as e:
        print(f"⚠ Failed to draw border basemap: {e}")


def build_manholes(c006, c05, crs=None):
    req_cols = ["node_name", "SH2S", "us_x", "us_y"]
    for c in req_cols:
        if c not in c006.columns:
            raise KeyError(f"c006 missing required column: {c}")

    agg = (
        c006.groupby("node_name")
        .agg(
            SH2S=("SH2S", "max"),
            us_x=("us_x", "first"),
            us_y=("us_y", "first")
        )
        .reset_index()
    )

    if "distance" in c006.columns:
        r006 = c006.groupby("node_name")["distance"].max().rename("r006")
        agg = agg.merge(r006, on="node_name", how="left")
    else:
        agg["r006"] = np.nan

    if ("node_name" in c05.columns) and ("distance" in c05.columns):
        r05 = c05.groupby("node_name")["distance"].max().rename("r05")
        agg = agg.merge(r05, on="node_name", how="left")
    else:
        agg["r05"] = np.nan

    geom = [Point(xy) for xy in zip(agg["us_x"], agg["us_y"])]
    return gpd.GeoDataFrame(agg, geometry=geom, crs=crs)


def build_pipes(c006, rp, crs=None):
    name_col = "name" if "name" in rp.columns else \
               "pipe_name" if "pipe_name" in rp.columns else None
    if name_col is None:
        raise KeyError("rp missing 'name' or 'pipe_name' column")

    coords = rp[[name_col, "ds_x", "ds_y"]].copy()
    coords.columns = ["pipe_name", "ds_x", "ds_y"]

    m = (
        c006.merge(coords, on="pipe_name", how="left")
        .dropna(subset=["us_x", "us_y", "ds_x", "ds_y"])
        .copy()
    )

    if "dcorr_dt" not in m.columns:
        m["dcorr_dt"] = np.nan

    geom = [
        LineString([(r.us_x, r.us_y), (r.ds_x, r.ds_y)])
        for _, r in m.iterrows()
    ]
    return gpd.GeoDataFrame(m, geometry=geom, crs=crs)


# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE CORROSION REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def report_maintenance_corrosion_percentage(pipes, maint_gdf, threshold=0.5,
                                            city_name="Hong Kong"):
    print("=" * 60)
    print(f"Maintenance project corrosion report – {city_name}")

    if pipes is None or len(pipes) == 0:
        print("⚠ No pipe geometries available.")
        return np.nan
    if maint_gdf is None or len(maint_gdf) == 0:
        print("⚠ No maintenance project geometry available.")
        return np.nan
    if "dcorr_dt" not in pipes.columns:
        print("⚠ pipes missing 'dcorr_dt'.")
        return np.nan

    pp = pipes.copy()
    mm = maint_gdf.copy()
    pp = pp[pp.geometry.notna() & ~pp.geometry.is_empty].copy()
    mm = mm[mm.geometry.notna() & ~mm.geometry.is_empty].copy()
    pp["dcorr_dt"] = pd.to_numeric(pp["dcorr_dt"], errors="coerce")
    pp = pp.dropna(subset=["dcorr_dt"]).copy()

    if len(pp) == 0 or len(mm) == 0:
        print("⚠ No valid pipe / maintenance geometries.")
        return np.nan

    if pp.crs is not None and mm.crs is not None and str(pp.crs) != str(mm.crs):
        mm = mm.to_crs(pp.crs)

    try:
        inter = gpd.overlay(
            pp[["dcorr_dt", "geometry"]],
            mm[["geometry"]],
            how="intersection",
            keep_geom_type=False
        )
    except Exception as e:
        print(f"⚠ geopandas.overlay failed: {e}")
        return np.nan

    if inter is None or len(inter) == 0:
        print("⚠ No overlap between sewer pipes and maintenance project.")
        return 0.0

    inter = inter[inter.geometry.notna() & ~inter.geometry.is_empty].copy()
    if len(inter) == 0:
        return 0.0

    inter["seg_len"] = inter.geometry.length
    inter = inter[np.isfinite(inter["seg_len"]) & (inter["seg_len"] > 0)].copy()
    if len(inter) == 0:
        return 0.0

    total_len = float(inter["seg_len"].sum())
    high_len = float(inter.loc[inter["dcorr_dt"] > threshold, "seg_len"].sum())
    pct = 100.0 * high_len / total_len if total_len > 0 else np.nan

    print(f"✓ Maintenance pipe length with corrosion > {threshold:g} mm/year: {pct:.2f}%")
    return pct


# ═══════════════════════════════════════════════════════════════════════════════
# WWTP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _clean_node_series(s):
    out = s.astype(str).str.strip()
    out = out.replace({
        "": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan,
        "NONE": np.nan, "null": np.nan, "NULL": np.nan
    })
    return out


def build_wwtp_from_rp(rp, crs=None):
    us_col = "start"
    ds_col = "end"
    required_cols = {us_col, ds_col, "ds_x", "ds_y"}
    missing = required_cols - set(rp.columns)

    if missing:
        print(f"⚠ rp missing required columns for WWTP detection: {missing}")
        return gpd.GeoDataFrame(
            columns=["node_name", "ds_x", "ds_y", "geometry"],
            geometry="geometry", crs=crs)

    tmp = rp[[us_col, ds_col, "ds_x", "ds_y"]].copy()
    tmp[us_col] = _clean_node_series(tmp[us_col])
    tmp[ds_col] = _clean_node_series(tmp[ds_col])
    tmp["ds_x"] = pd.to_numeric(tmp["ds_x"], errors="coerce")
    tmp["ds_y"] = pd.to_numeric(tmp["ds_y"], errors="coerce")
    tmp = tmp.dropna(subset=[ds_col, "ds_x", "ds_y"])

    upstream_nodes = set(tmp[us_col].dropna().astype(str))
    downstream_nodes = set(tmp[ds_col].dropna().astype(str))
    wwtp_nodes = downstream_nodes - upstream_nodes
    wwtp_df = tmp[tmp[ds_col].isin(wwtp_nodes)].copy()

    if wwtp_df.empty:
        print("  Found 0 WWTP nodes")
        return gpd.GeoDataFrame(
            columns=["node_name", "ds_x", "ds_y", "geometry"],
            geometry="geometry", crs=crs)

    wwtp_df = (
        wwtp_df[[ds_col, "ds_x", "ds_y"]]
        .drop_duplicates(subset=[ds_col])
        .rename(columns={ds_col: "node_name"})
        .copy()
    )
    geom = [Point(xy) for xy in zip(wwtp_df["ds_x"], wwtp_df["ds_y"])]
    wwtp_gdf = gpd.GeoDataFrame(wwtp_df, geometry=geom, crs=crs)
    print(f"  Found {len(wwtp_gdf):,} WWTP nodes")
    return wwtp_gdf


def plot_wwtp(ax, wwtp, show_label=False):
    if wwtp is None or len(wwtp) == 0:
        return
    wwtp.plot(
        ax=ax, marker="^", color=WWTP_COLOR, markersize=WWTP_MARKER_SIZE,
        alpha=0.95, edgecolor="none", zorder=9,
        label="WWTP" if show_label else None
    )


def auto_prepare_city_spatial_data(cfg, rp, c006, c05):
    x_cands, y_cands = [], []
    for col in ["us_x", "ds_x"]:
        if col in c006.columns:
            x_cands.append(c006[col])
        if col in rp.columns:
            x_cands.append(rp[col])
    for col in ["us_y", "ds_y"]:
        if col in c006.columns:
            y_cands.append(c006[col])
        if col in rp.columns:
            y_cands.append(rp[col])

    x_all = pd.concat(x_cands, ignore_index=True) if x_cands else pd.Series(dtype=float)
    y_all = pd.concat(y_cands, ignore_index=True) if y_cands else pd.Series(dtype=float)

    src_crs = cfg.get("source_crs", None)
    plot_crs = cfg.get("plot_crs", None)

    if src_crs is None:
        if detect_lonlat_from_xy(x_all, y_all):
            src_crs = "EPSG:4326"
            print(f"  [{cfg['name']}] source CRS auto-detected as EPSG:4326")
        else:
            src_crs = None

    if plot_crs is None:
        if src_crs is not None and CRS.from_user_input(src_crs).is_geographic:
            plot_crs = choose_local_projected_crs_from_xy(x_all, y_all, cfg["name"])
        else:
            plot_crs = src_crs

    mh = build_manholes(c006, c05, crs=src_crs)
    pipes = build_pipes(c006, rp, crs=src_crs)
    wwtp = build_wwtp_from_rp(rp, crs=src_crs)

    if plot_crs is not None and mh.crs is not None and str(mh.crs) != str(plot_crs):
        mh = mh.to_crs(plot_crs)
    if plot_crs is not None and pipes.crs is not None and str(pipes.crs) != str(plot_crs):
        pipes = pipes.to_crs(plot_crs)
    if (plot_crs is not None and wwtp is not None and len(wwtp) > 0
            and wwtp.crs is not None and str(wwtp.crs) != str(plot_crs)):
        wwtp = wwtp.to_crs(plot_crs)

    print(f"  [{cfg['name']}] final plot CRS = {plot_crs}")
    return mh, pipes, wwtp, src_crs, plot_crs


def get_data_extent(mh, pipes, wwtp=None, pad=0.05):
    bounds_list = []
    if pipes is not None and len(pipes) > 0:
        bounds_list.append(pipes.total_bounds)
    if mh is not None and len(mh) > 0:
        bounds_list.append(mh.total_bounds)
    if wwtp is not None and len(wwtp) > 0:
        bounds_list.append(wwtp.total_bounds)
    if not bounds_list:
        return (0, 1), (0, 1)

    xmin = min(b[0] for b in bounds_list)
    ymin = min(b[1] for b in bounds_list)
    xmax = max(b[2] for b in bounds_list)
    ymax = max(b[3] for b in bounds_list)
    dx, dy = (xmax - xmin) or 1.0, (ymax - ymin) or 1.0
    return (xmin - dx * pad, xmax + dx * pad), \
           (ymin - dy * pad, ymax + dy * pad)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _add_panel_label(ax, label):
    if not label:
        return
    ax.text(
        0.0, 1.02, label, transform=ax.transAxes,
        fontsize=10, fontweight="bold", ha="left", va="bottom", color=TEXT_C
    )


def _fmt_map(ax, title, xlim=None, ylim=None):
    apply_axis_style(ax, facecolor=BG_AX, half_frame=False)
    ax.set_title(title, fontsize=8, fontweight="normal", loc="center",
                 pad=6, color=TEXT_C)
    ax.tick_params(labelsize=5, direction="in", length=2.2, width=0.4, colors=TEXT_C)
    ax.set_aspect("equal", adjustable="box")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x / 1000:.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y / 1000:.0f}"))
    ax.set_xlabel("Easting (km)", fontsize=6, color=TEXT_C, labelpad=1.5)
    ax.set_ylabel("Northing (km)", fontsize=6, color=TEXT_C, labelpad=1.5)


def _style_legend(leg, facecolor="white"):
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor(facecolor)
    frame.set_alpha(0.95)
    frame.set_edgecolor("none")


def _scalebar(ax, length_m=None, plot_crs=None):
    if not is_metric_crs(plot_crs):
        return
    xl, yl = ax.get_xlim(), ax.get_ylim()
    if length_m is None:
        length_m = auto_scalebar_length_m(xl)
    if length_m is None or length_m <= 0:
        return

    x0 = xl[1] - (xl[1] - xl[0]) * 0.04 - length_m
    y0 = yl[0] + (yl[1] - yl[0]) * 0.025
    dy = (yl[1] - yl[0]) * 0.005

    ax.plot([x0, x0 + length_m], [y0, y0], color=TEXT_C, lw=1.8, zorder=10, clip_on=False)
    for xp in (x0, x0 + length_m):
        ax.plot([xp, xp], [y0 - dy, y0 + dy], color=TEXT_C, lw=1.2, zorder=10, clip_on=False)

    km = length_m / 1000
    label = f"{km:.0f} km" if abs(km - round(km)) < 1e-9 else f"{km:.1f} km"
    ax.text(x0 + length_m / 2, y0 + dy * 3.5, label, ha="center", fontsize=5,
            fontweight="bold", color=TEXT_C, zorder=10)


def _north_arrow(ax):
    xl, yl = ax.get_xlim(), ax.get_ylim()
    x = xl[1] - (xl[1] - xl[0]) * 0.04
    y = yl[1] - (yl[1] - yl[0]) * 0.06
    dy = (yl[1] - yl[0]) * 0.035
    ax.annotate("", xy=(x, y), xytext=(x, y - dy),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=TEXT_C), zorder=10)
    ax.text(x, y + dy * 0.15, "N", ha="center", va="bottom", fontsize=7,
            fontweight="bold", color=TEXT_C, zorder=10)


def _make_radius_circles(gdf, radius_col, scale=1.0):
    """生成散点圆 patch；scale 用于放大半径（figure d/g/j）。"""
    patches = []
    if gdf is None or len(gdf) == 0:
        return patches
    for _, r in gdf.iterrows():
        radius = r.get(radius_col, np.nan)
        if not np.isfinite(radius) or radius <= 0:
            continue
        geom = r.geometry
        if geom is None or geom.is_empty:
            continue
        patches.append(Circle((geom.x, geom.y), float(radius) * float(scale)))
    return patches


# ═══════════════════════════════════════════════════════════════════════════════
# MAP ROW PANELS
# ═══════════════════════════════════════════════════════════════════════════════
def _drop_top_radius_outliers(gdf, radius_col, outlier_percent=0.0,
                              city_name="", radius_label=""):
    if gdf is None or len(gdf) == 0:
        return gdf

    try:
        pct = float(outlier_percent)
    except Exception:
        pct = 0.0
    if pct <= 0:
        return gdf

    pct = min(pct, 100.0)
    vals = pd.to_numeric(gdf[radius_col], errors="coerce")
    valid = vals.notna() & np.isfinite(vals) & (vals > 0)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return gdf

    cutoff = float(np.nanpercentile(vals.loc[valid].values, 100.0 - pct))
    keep = valid & (vals <= cutoff)
    removed = int(n_valid - keep.sum())
    if removed > 0:
        print(
            f"  [{city_name}] panel {radius_label}: removed {removed:,}/"
            f"{n_valid:,} top-radius outliers "
            f"({pct:g}%, cutoff={cutoff:.4g})"
        )
    return gdf.loc[keep].copy()


def plot_map_row(axes, mh, pipes, wwtp, maint_gdf, border_gdf,
                 xlim, ylim, labels, city_name, plot_crs, is_hk=False,
                 scalebar_m=None, show_legend=True, disp_radius_scale=1.0,
                 disp_outlier_percent=0.0):
    ax1, ax2, ax3 = axes
    legend_loc = "lower left"

    for _ax in (ax1, ax2, ax3):
        draw_border_basemap(_ax, border_gdf)

    # ── Panel 1: H2S concentration ──
    mh1 = mh.copy()
    if "SH2S" not in mh1.columns:
        mh1["SH2S"] = np.nan
    mh1["cat"] = pd.cut(mh1["SH2S"], bins=H2S_BINS, labels=H2S_LABELS,
                        right=True, include_lowest=True)

    pipes.plot(ax=ax1, color=BG_PIPE_CLR, linewidth=0.22, alpha=0.55, zorder=1)
    ms_list = H2S_MS_HK if is_hk else H2S_MS_DEF

    for i, lab in enumerate(H2S_LABELS):
        sub = mh1[mh1["cat"] == lab]
        if sub.empty:
            continue
        sub.plot(ax=ax1, color=H2S_COLORS[i], markersize=ms_list[i],
                 alpha=0.88, edgecolor="none", zorder=3 + i)

    plot_wwtp(ax1, wwtp, show_label=False)

    if show_legend:
        h1 = [Line2D([], [], marker="o", ls="", color=H2S_COLORS[i],
                     markersize=3.3, label=f"{lab} ppm")
              for i, lab in enumerate(H2S_LABELS)]
        h1.append(Line2D([], [], marker="^", ls="", color=WWTP_COLOR,
                         markerfacecolor=WWTP_COLOR, markeredgecolor=WWTP_COLOR,
                         markersize=4.2, label="WWTP"))
        leg = ax1.legend(handles=h1, title=r"$\mathrm{H_2S}$ conc. (ppm)",
                         loc=legend_loc, frameon=True, framealpha=0.95,
                         edgecolor="none", title_fontsize=6, fontsize=5.4)
        _style_legend(leg, facecolor="white")

    _fmt_map(ax1, city_name + r" – $\mathrm{H_2S}$ concentration", xlim, ylim)
    _add_panel_label(ax1, labels[0])
    _scalebar(ax1, length_m=scalebar_m, plot_crs=plot_crs)
    _north_arrow(ax1)

    # ── Panel 2: Atmospheric dispersion (figure d / g / j) ──
    pipes.plot(ax=ax2, color=DISP_PIPE_CLR, linewidth=DISP_PIPE_LW,
               alpha=DISP_PIPE_ALPHA, zorder=1)

    sel006 = (mh.dropna(subset=["r006"]).query("r006 > 0").copy()
              if "r006" in mh.columns else mh.iloc[0:0].copy())
    sel05 = (mh.dropna(subset=["r05"]).query("r05 > 0").copy()
             if "r05" in mh.columns else mh.iloc[0:0].copy())

    # >>> 这里把散点圆半径乘以 disp_radius_scale（10 或 20 倍） <<<
    sel006 = _drop_top_radius_outliers(
        sel006, "r006", outlier_percent=disp_outlier_percent,
        city_name=city_name, radius_label=labels[1] + " r006"
    )
    sel05 = _drop_top_radius_outliers(
        sel05, "r05", outlier_percent=disp_outlier_percent,
        city_name=city_name, radius_label=labels[1] + " r05"
    )
    patches006 = _make_radius_circles(sel006, "r006", scale=disp_radius_scale)
    patches05 = _make_radius_circles(sel05, "r05", scale=disp_radius_scale)

    if len(patches006):
        ax2.add_collection(PatchCollection(
            patches006, facecolor=DISP_CLR_01, edgecolor="none",
            alpha=DISP_ALPHA_01, linewidth=0, zorder=2))
    if len(patches05):
        ax2.add_collection(PatchCollection(
            patches05, facecolor=DISP_CLR_001, edgecolor="none",
            alpha=DISP_ALPHA_001, linewidth=0, zorder=3))

    plot_wwtp(ax2, wwtp, show_label=False)

    if show_legend:
        scale_note = (f" (×{disp_radius_scale:g})"
                      if abs(disp_radius_scale - 1.0) > 1e-9 else "")
        h2 = [
            Line2D([], [], color=DISP_PIPE_CLR, lw=0.8, alpha=DISP_PIPE_ALPHA,
                   label="Sewer pipes"),
            mpatches.Patch(fc=DISP_CLR_01, alpha=DISP_LEGEND_ALPHA_01, ec="none",
                           label=r"$\mathrm{H_2S}$ > 0.1 ppm zone" + scale_note),
            mpatches.Patch(fc=DISP_CLR_001, alpha=DISP_LEGEND_ALPHA_001, ec="none",
                           label=r"$\mathrm{H_2S}$ > 0.01 ppm zone" + scale_note),
            Line2D([], [], marker="^", ls="", color=WWTP_COLOR,
                   markerfacecolor=WWTP_COLOR, markeredgecolor=WWTP_COLOR,
                   markersize=4.2, label="WWTP")
        ]
        leg = ax2.legend(handles=h2, loc=legend_loc, frameon=True,
                         framealpha=0.95, edgecolor="none", fontsize=5.4)
        _style_legend(leg, facecolor="white")

    _fmt_map(ax2, f"{city_name} – Atmospheric dispersion", xlim, ylim)
    _add_panel_label(ax2, labels[1])
    _scalebar(ax2, length_m=scalebar_m, plot_crs=plot_crs)
    _north_arrow(ax2)

    # ── Panel 3: corrosion rate ──
    pp = pipes.copy()
    if "dcorr_dt" not in pp.columns:
        pp["dcorr_dt"] = np.nan
    pp["ccat"] = pd.cut(pp["dcorr_dt"].clip(lower=0), bins=CORR_BINS,
                        labels=CORR_LABELS, right=True, include_lowest=True)

    pipes.plot(ax=ax3, color="#D9D4CB", linewidth=0.16, alpha=0.35, zorder=1)

    for i, lab in enumerate(CORR_LABELS):
        sub = pp[pp["ccat"] == lab]
        if sub.empty:
            continue
        sub.plot(ax=ax3, color=CORR_COLORS[i], linewidth=CORR_LWS[i],
                 alpha=0.88, zorder=2 + i)

    plot_wwtp(ax3, wwtp, show_label=False)

    if show_legend:
        h3 = [Line2D([], [], color=CORR_COLORS[i], lw=CORR_LWS[i] * 1.8,
                     label=f"{lab}" + r" mm yr$^{-1}$")
              for i, lab in enumerate(CORR_LABELS)]
        h3.append(Line2D([], [], marker="^", ls="", color=WWTP_COLOR,
                         markerfacecolor=WWTP_COLOR, markeredgecolor=WWTP_COLOR,
                         markersize=4.2, label="WWTP"))
        leg = ax3.legend(handles=h3, title="Corrosion rate", loc=legend_loc,
                         frameon=True, framealpha=0.95, edgecolor="none",
                         title_fontsize=6, fontsize=5.4)
        _style_legend(leg, facecolor="white")

    _fmt_map(ax3, f"{city_name} – Corrosion rate", xlim, ylim)
    _add_panel_label(ax3, labels[2])
    _scalebar(ax3, length_m=scalebar_m, plot_crs=plot_crs)
    _north_arrow(ax3)


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER EXTRACTION & DISTRIBUTION PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════
def remove_outliers_percentile(data, lower=5, upper=95):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return data
    p_low, p_high = np.percentile(data, [lower, upper])
    return data[(data >= p_low) & (data <= p_high)]


def extract_pipe_params(rp_df, city_name):
    params = {}
    df = rp_df.copy()

    if city_name == "Hong Kong":
        pipe_col = None
        for col in ["pipe_name", "name"]:
            if col in df.columns:
                pipe_col = col
                break
        if pipe_col is not None:
            before_n = len(df)
            mask = ~df[pipe_col].astype(str).str.startswith("Link_", na=False)
            df = df.loc[mask].copy()
            print(f"    Filtered HK Link_ pipes using '{pipe_col}': {before_n:,} -> {len(df):,}")

    candidates = {
        "A/V": ["A_V", "a_v", "AV", "av", "A/V"],
        "Velocity": ["v", "V", "velocity", "Velocity", "VELOCITY"],
        "Flowrate": ["flowrate", "Flowrate", "FLOWRATE", "flow", "Q"],
        "Pipe diameter": ["WIDTH", "width", "diameter", "DIAMETER", "Diameter", "DIAM"],
        "Slope": ["slope", "SLOPE", "gradient", "GRADIENT", "Slope", "Gradient"],
    }

    for pname, cols in candidates.items():
        for col in cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna().values
                if len(vals) > 0:
                    if pname == "Pipe diameter":
                        vals = vals * 1000   # m → mm
                    params[pname] = vals
                    print(f"    {pname}: column='{col}', N={len(vals)}")
                break

    depth_col, diam_col = None, None
    for col in ["depth", "Depth", "DEPTH", "flow_depth"]:
        if col in df.columns:
            depth_col = col
            break
    for col in ["diameter", "DIAMETER", "Diameter", "WIDTH", "width"]:
        if col in df.columns:
            diam_col = col
            break

    if depth_col and diam_col:
        d = pd.to_numeric(df[depth_col], errors="coerce")
        w = pd.to_numeric(df[diam_col], errors="coerce")
        fr = (d / w).replace([np.inf, -np.inf], np.nan)
        mask = fr.notna() & (fr > 0) & (fr <= 1.5)
        if mask.sum() > 0:
            params["Filling ratio"] = fr[mask].values
            print(f"    Filling ratio: N={mask.sum()}")

    return params


def make_param_legend_handles(city_order, city_colors):
    handles = [
        mpatches.Patch(fc=city_colors[c], alpha=0.45, ec=SPINE_C, lw=0.6, label=c)
        for c in city_order
    ]
    handles += [
        Line2D([], [], color=ACCENT_RED, lw=1.4, label="Median"),
        Line2D([], [], marker="D", ls="", markerfacecolor="white",
               markeredgecolor=SPINE_C, markersize=3.5, markeredgewidth=0.5,
               label="Mean"),
        mpatches.Patch(fc=SCI_WHISKER_BAND_COLOR, alpha=SCI_WHISKER_BAND_ALPHA,
                       ec="none", label="SCI whisker range"),
    ]
    return handles


def plot_param_panels(axes, all_city_params, param_order, units, city_order,
                      city_colors, panel_labels, sci_whisker_ranges=None,
                      strip_sample_ratio=0.10, min_strip_pts=50):
    rng = np.random.default_rng(42)
    sci_whisker_ranges = sci_whisker_ranges or {}

    for idx, (ax, pname, unit, plbl) in enumerate(
            zip(axes, param_order, units, panel_labels)):
        apply_axis_style(ax, facecolor=BG_AX, half_frame=True)

        data_list, colors_list, city_labels = [], [], []
        for city in city_order:
            if city in all_city_params and pname in all_city_params[city]:
                raw = all_city_params[city][pname]
                clean = remove_outliers_percentile(raw, lower=5, upper=95)
                if len(clean) > 0:
                    data_list.append(clean)
                    colors_list.append(city_colors[city])
                    city_labels.append(city)

        if len(data_list) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center",
                    va="center", fontsize=7, color=SUBTEXT_C)
            _add_panel_label(ax, plbl)
            continue

        positions = list(range(len(data_list)))

        if pname in sci_whisker_ranges:
            wmin, wmax = sci_whisker_ranges[pname]
            if np.isfinite(wmin) and np.isfinite(wmax) and wmax > wmin:
                ax.axhspan(wmin, wmax, color=SCI_WHISKER_BAND_COLOR,
                           alpha=SCI_WHISKER_BAND_ALPHA, zorder=0.5, linewidth=0)

        bp = ax.boxplot(
            data_list, positions=positions, vert=True, widths=0.50,
            patch_artist=True, showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor=SPINE_C, markersize=3.5,
                           markeredgewidth=0.5),
            medianprops=dict(color=ACCENT_RED, linewidth=1.4),
            whiskerprops=dict(color=SPINE_C, linewidth=0.8),
            capprops=dict(color=SPINE_C, linewidth=0.8),
            flierprops=dict(marker="", linewidth=0),
            zorder=3
        )

        for patch, clr in zip(bp["boxes"], colors_list):
            patch.set_facecolor(clr)
            patch.set_alpha(0.45)
            patch.set_edgecolor(SPINE_C)
            patch.set_linewidth(0.6)

        for i, (d, clr) in enumerate(zip(data_list, colors_list)):
            n_total = len(d)
            n_sample = max(min_strip_pts, int(np.ceil(n_total * strip_sample_ratio)))
            n_sample = min(n_sample, n_total)
            strip = rng.choice(d, size=n_sample, replace=False) \
                if n_sample < n_total else d
            jitter = rng.uniform(-0.17, 0.17, size=len(strip))
            ax.scatter(i + jitter, strip, s=1.5, alpha=0.16, color=clr,
                       edgecolors="none", zorder=2, rasterized=True)

        ylabel = f"{pname} [{unit}]" if unit else pname
        ax.set_ylabel(ylabel, fontsize=7, color=TEXT_C)

        abbrev_map = {"Hong Kong": "HK", "Toronto": "Toronto", "Los Angeles": "LA"}
        ax.set_xticks(positions)
        ax.set_xticklabels([abbrev_map.get(c, c) for c in city_labels],
                           fontsize=6, rotation=0, color=TEXT_C)
        ax.tick_params(axis="y", labelsize=6, colors=TEXT_C)
        ax.grid(axis="y", color=GRID_C, linestyle=":", linewidth=0.5, alpha=0.7)

        _add_panel_label(ax, plbl)


# ═══════════════════════════════════════════════════════════════════════════════
# WWTP PREDICTED vs MEASURED (code 2)
# ═══════════════════════════════════════════════════════════════════════════════
def find_flowrate_col(df):
    for c in ["flowrate", "Flowrate", "FLOWRATE", "flow", "Q", "flow_rate"]:
        if c in df.columns:
            return c
    raise KeyError(f"No flowrate column found. Columns are: {list(df.columns)}")


def find_end_col(df):
    for c in ["end", "End", "END", "ds_node", "downstream", "end_node"]:
        if c in df.columns:
            return c
    raise KeyError(f"No 'end' column found. Columns are: {list(df.columns)}")


def load_measured(xlsx_path):
    if not os.path.exists(xlsx_path):
        print(f"⚠ Measured Excel not found: {xlsx_path}")
        print("  Using hard-coded fallback measured values.")
        return dict(MEASURED_FALLBACK)
    try:
        df = pd.read_excel(xlsx_path, sheet_name="WWTP flow")
    except Exception as e:
        print(f"⚠ Could not read Excel sheet 'WWTP flow': {e}")
        return dict(MEASURED_FALLBACK)

    name_col = None
    val_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "catchment name":
            name_col = c
        if "wwtp flow" in cl:
            val_col = c

    if name_col is None or val_col is None:
        try:
            name_col = df.columns[1]
            val_col = df.columns[3]
            print(f"  Falling back to columns: name='{name_col}', value='{val_col}'")
        except Exception:
            print("⚠ Could not locate measured columns. Using fallback dict.")
            return dict(MEASURED_FALLBACK)

    out = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        val = pd.to_numeric(row[val_col], errors="coerce")
        if name and name.lower() != "nan" and np.isfinite(val):
            out[name] = float(val)

    if not out:
        print("⚠ Measured dict empty after parsing. Using fallback.")
        return dict(MEASURED_FALLBACK)

    print(f"✓ Loaded {len(out)} measured WWTP values from Excel.")
    return out


def predicted_inflow_for_city(rp_csv, node_to_catchment, factor):
    df = pd.read_csv(rp_csv)
    end_col = find_end_col(df)
    flow_col = find_flowrate_col(df)
    df[end_col] = df[end_col].astype(str).str.strip()
    df[flow_col] = pd.to_numeric(df[flow_col], errors="coerce")
    grouped = df.groupby(end_col)[flow_col].sum()

    out = {}
    for node in node_to_catchment:
        if node in grouped.index:
            out[node] = float(grouped.loc[node]) * factor
        else:
            print(f"  ⚠ end node '{node}' not found in {os.path.basename(rp_csv)}")
            out[node] = np.nan
    return out


def build_comparison():
    measured = load_measured(MEASURED_XLSX)
    rows = []
    for city, (rp_csv, node_map) in CITY_NODE_MAPS.items():
        print("=" * 60)
        print(f"Processing WWTP inflow – {city} …")
        if not os.path.exists(rp_csv):
            print(f"⚠ result_segments file not found: {rp_csv} — skipping {city}.")
            continue
        pred = predicted_inflow_for_city(rp_csv, node_map, FLOW_UNIT_FACTOR)
        for node, catchment in node_map.items():
            p = pred.get(node, np.nan)
            m = measured.get(catchment, np.nan)
            if not np.isfinite(m):
                print(f"  ⚠ No measured value for '{catchment}' (node {node}).")
            rows.append({
                "city": city, "node": node, "plant": catchment,
                "predicted_MLd": p, "measured_MLd": m,
            })
    return pd.DataFrame(rows)


def compute_stats(pred, meas):
    pred = np.asarray(pred, dtype=float)
    meas = np.asarray(meas, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(meas)
    pred, meas = pred[mask], meas[mask]

    stats = {"n": len(pred)}
    if len(pred) < 2:
        return stats

    ss_res = np.sum((pred - meas) ** 2)
    ss_tot = np.sum((meas - meas.mean()) ** 2)
    stats["R2_linear"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    pos = (pred > 0) & (meas > 0)
    if pos.sum() >= 2:
        lp, lm = np.log10(pred[pos]), np.log10(meas[pos])
        ss_res_l = np.sum((lp - lm) ** 2)
        ss_tot_l = np.sum((lm - lm.mean()) ** 2)
        stats["R2_log"] = 1 - ss_res_l / ss_tot_l if ss_tot_l > 0 else np.nan

    stats["RMSE"] = np.sqrt(np.mean((pred - meas) ** 2))

    mean_meas = np.mean(meas)
    stats["nRMSE"] = (stats["RMSE"] / mean_meas * 100) if mean_meas != 0 else np.nan

    nz = meas != 0
    stats["MAPE"] = np.mean(np.abs((pred[nz] - meas[nz]) / meas[nz])) * 100
    return stats


def plot_wwtp_comparison(ax, res, label="a", use_log=True):
    ax.set_facecolor(BG_AX)

    valid = res.dropna(subset=["predicted_MLd", "measured_MLd"]).copy()
    if use_log:
        valid = valid[(valid["predicted_MLd"] > 0) & (valid["measured_MLd"] > 0)]

    if valid.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color=SUBTEXT_C)
        _add_panel_label(ax, label)
        return None

    stats = compute_stats(valid["predicted_MLd"], valid["measured_MLd"])

    all_vals = np.concatenate(
        [valid["measured_MLd"].values, valid["predicted_MLd"].values])

    if use_log:
        vmin = max(all_vals.min() * 0.5, 1e-3)
        vmax = all_vals.max() * 2.0
        line = np.array([vmin, vmax])
    else:
        vmin = 0.0
        vmax = all_vals.max() * 1.1
        line = np.array([vmin, vmax])

    ax.plot(line, line, color=SPINE_C, lw=1.0, ls="-", zorder=1, label="1:1 line")

    if use_log:
        ax.plot(line, line * 2, color="#999999", lw=0.7, ls="--", zorder=1,
                label="±2× range")
        ax.plot(line, line / 2, color="#999999", lw=0.7, ls="--", zorder=1)

    for city, sub in valid.groupby("city"):
        ax.scatter(sub["measured_MLd"], sub["predicted_MLd"], s=40,
                   color=CITY_COLORS.get(city, "#555555"), edgecolor=SPINE_C,
                   linewidth=0.6, alpha=0.9, zorder=3, label=city)

    if use_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    ax.set_xlabel("Measured WWTP inflow [ML d$^{-1}$]", color=TEXT_C, fontsize=6)
    ax.set_ylabel("Predicted WWTP inflow [ML d$^{-1}$]", color=TEXT_C, fontsize=6)

    grid_which = "both" if use_log else "major"
    ax.grid(True, which=grid_which, color=GRID_C, ls=":", lw=0.5, alpha=0.7)
    for s in ax.spines.values():
        s.set_color(SPINE_C)
        s.set_linewidth(0.7)
    ax.tick_params(colors=TEXT_C, labelsize=6)

    txt_lines = [f"n = {stats.get('n', 0)}"]
    if use_log and "R2_log" in stats and np.isfinite(stats["R2_log"]):
        txt_lines.append(f"$R^2_{{log}}$ = {stats['R2_log']:.3f}")
    if "R2_linear" in stats and np.isfinite(stats["R2_linear"]):
        txt_lines.append(f"$R^2$ = {stats['R2_linear']:.3f}")
    if "RMSE" in stats and np.isfinite(stats["RMSE"]):
        txt_lines.append(f"RMSE = {stats['RMSE']:.1f} ML/d")
    if use_log:
        if "MAPE" in stats and np.isfinite(stats["MAPE"]):
            txt_lines.append(f"MAPE = {stats['MAPE']:.1f}%")
    else:
        if "nRMSE" in stats and np.isfinite(stats["nRMSE"]):
            txt_lines.append(f"nRMSE = {stats['nRMSE']:.1f}%")

    ax.text(0.04, 0.96, "\n".join(txt_lines), transform=ax.transAxes,
            ha="left", va="top", fontsize=6, color=TEXT_C,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC",
                      lw=0.6, alpha=0.9), zorder=5)

    leg = ax.legend(loc="lower right", fontsize=5.4, frameon=True,
                    framealpha=0.95, edgecolor="none")
    leg.get_frame().set_facecolor("white")

    scale_tag = "log–log" if use_log else "linear"
    ax.set_title(f"Predicted vs measured WWTP inflow ({scale_tag})",
                 color=TEXT_C, fontsize=8, pad=6)
    _add_panel_label(ax, label)

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# DATA EXPORT FOR THE RESULTS-CALCULATOR SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════
def export_figure2_data(out_dir, all_city_params, sci_whisker_ranges,
                        city_mh, city_pipes, wwtp_res, segment_counts):
    """Export every dataset needed to reproduce the Results numbers."""
    ensure_dir(out_dir)
    print("=" * 60)
    print(f"Exporting Figure 2 data → {out_dir}")

    # 1. WWTP predicted vs measured
    wwtp_res.to_csv(os.path.join(out_dir, "wwtp_comparison.csv"), index=False)

    # 2. pipe hydraulic / geometric parameters (long format: city, parameter, value)
    frames = []
    for city, params in all_city_params.items():
        for pname, vals in params.items():
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            frames.append(pd.DataFrame({"city": city,
                                        "parameter": pname,
                                        "value": vals}))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(out_dir, "pipe_parameters_long.csv"), index=False)

    # 3. SCI whisker ranges
    sci_rows = [{"parameter": k, "whisker_min": v[0], "whisker_max": v[1]}
                for k, v in (sci_whisker_ranges or {}).items()]
    pd.DataFrame(sci_rows).to_csv(
        os.path.join(out_dir, "sci_whisker_ranges.csv"), index=False)

    # 4. nodes: H2S concentration + dispersion radii
    node_frames = []
    for city, mh in city_mh.items():
        if mh is None or len(mh) == 0:
            continue
        node_frames.append(pd.DataFrame({
            "city": city,
            "node_name": (mh["node_name"].values if "node_name" in mh.columns
                          else np.arange(len(mh))),
            "SH2S": (pd.to_numeric(mh["SH2S"], errors="coerce").values
                     if "SH2S" in mh.columns else np.nan),
            "r006_0p1ppm": (pd.to_numeric(mh["r006"], errors="coerce").values
                            if "r006" in mh.columns else np.nan),
            "r05_0p01ppm": (pd.to_numeric(mh["r05"], errors="coerce").values
                            if "r05" in mh.columns else np.nan),
        }))
    if node_frames:
        pd.concat(node_frames, ignore_index=True).to_csv(
            os.path.join(out_dir, "nodes.csv"), index=False)

    # 5. pipe corrosion rate + segment length (metres, from projected geometry)
    pipe_frames = []
    for city, pp in city_pipes.items():
        if pp is None or len(pp) == 0:
            continue
        try:
            lengths = pp.geometry.length.values
        except Exception:
            lengths = np.full(len(pp), np.nan)
        pipe_frames.append(pd.DataFrame({
            "city": city,
            "dcorr_dt": (pd.to_numeric(pp["dcorr_dt"], errors="coerce").values
                         if "dcorr_dt" in pp.columns else np.nan),
            "length_m": lengths,
        }))
    if pipe_frames:
        pd.concat(pipe_frames, ignore_index=True).to_csv(
            os.path.join(out_dir, "pipes_corrosion.csv"), index=False)

    # 6. number of mapped pipe segments per city
    pd.DataFrame([{"city": c, "n_segments": int(n)}
                  for c, n in segment_counts.items()]).to_csv(
        os.path.join(out_dir, "segment_counts.csv"), index=False)

    print("✓ Exported: wwtp_comparison.csv, pipe_parameters_long.csv, "
          "sci_whisker_ranges.csv, nodes.csv, pipes_corrosion.csv, "
          "segment_counts.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def _ax_rect(fig, xrange, yrange):
    l, r = xrange
    b, t = yrange
    return fig.add_axes([l, b, r - l, t - b])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    args = parse_args()
    results_root = args.results_root.resolve()
    OUTPUT_DIR = str(args.output_dir.resolve())
    FIG2_DATA_DIR = os.path.join(OUTPUT_DIR, "figure2_data")
    SCI_WHISKER_CSV = str(
        (args.whisker_csv or results_root / "figure1" / "hk_whisker_ranges.csv").resolve()
    )
    city_dirs = {
        "Hong Kong": results_root / "HK_v3" / "biochemical_results",
        "Toronto": results_root / "toronto_v3" / "biochemical_results",
        "Los Angeles": results_root / "LA_v3" / "biochemical_results",
    }
    for config, city, prefix in (
        (HK_CFG, "Hong Kong", "hk"),
        (TOR_CFG, "Toronto", "toronto"),
        (LA_CFG, "Los Angeles", "la"),
    ):
        bio_dir = city_dirs[city]
        config["rp_csv"] = str(bio_dir / f"{prefix}_result_segments_v7.csv")
        config["c006_csv"] = str(bio_dir / f"{prefix}_emission_radius_segments_v7.csv")
        config["c05_csv"] = str(bio_dir / f"{prefix}_emission_radius_odour_segments_v7.csv")

    # CITY_NODE_MAPS is created at module import time, so refresh its CSV paths
    # after applying the command-line results root. This is essential on Linux
    # and also prevents Figure 2 from silently reading an older Windows result.
    CITY_NODE_MAPS = {
        "Hong Kong": (HK_CFG["rp_csv"], HK_NODE_TO_CATCHMENT),
        "Toronto": (TOR_CFG["rp_csv"], TOR_NODE_TO_CATCHMENT),
        "Los Angeles": (LA_CFG["rp_csv"], LA_NODE_TO_CATCHMENT),
    }

    ensure_dir(OUTPUT_DIR)

    CITY_ORDER = ["Hong Kong", "Toronto", "Los Angeles"]
    PARAM_ORDER = ["A/V", "Velocity", "Flowrate",
                   "Pipe diameter", "Slope", "Filling ratio"]
    PARAM_UNITS = ["1/m", "m/s", "m³/s", "mm", "-", "-"]
    PARAM_LABELS = ["b", "", "", "", "", ""]

    print(f"✓ DISP_RADIUS_SCALE = {DISP_RADIUS_SCALE} "
          f"(figure d/g/j 散点半径放大倍数)")

    print(f"DISP_RADIUS_OUTLIER_PERCENT = {DISP_RADIUS_OUTLIER_PERCENT:g}% "
          "(figure d/g/j top-radius trimming)")

    if not LAYOUT_PREVIEW_ONLY:
        print("=" * 60)
        print("Loading SCI whisker ranges …")
        sci_whisker_ranges = load_sci_whisker_ranges(SCI_WHISKER_CSV)

        print("=" * 60)
        print("Loading Hong Kong …")
        hk_rp, hk_c006, hk_c05 = load_csvs(HK_CFG)
        hk_mh, hk_pipe, hk_wwtp, hk_src_crs, hk_plot_crs = auto_prepare_city_spatial_data(
            HK_CFG, hk_rp, hk_c006, hk_c05)
        hk_border = load_border(HK_CFG, dst_crs=hk_plot_crs)
        hk_maint = load_shapefile(HK_CFG["maint_shp"], src_crs=HK_CFG["source_crs"],
                                  dst_crs=hk_plot_crs)
        hk_maint_high_corr_pct = report_maintenance_corrosion_percentage(
            pipes=hk_pipe, maint_gdf=hk_maint, threshold=0.5, city_name="Hong Kong")
        hk_xl, hk_yl = get_data_extent(hk_mh, hk_pipe, hk_wwtp, pad=0.05)
        if HK_CFG["xlim"] is not None:
            hk_xl = HK_CFG["xlim"]
        if HK_CFG["ylim"] is not None:
            hk_yl = HK_CFG["ylim"]

        print("=" * 60)
        print("Loading Toronto …")
        tor_rp, tor_c006, tor_c05 = load_csvs(TOR_CFG)
        tor_mh, tor_pipe, tor_wwtp, tor_src_crs, tor_plot_crs = auto_prepare_city_spatial_data(
            TOR_CFG, tor_rp, tor_c006, tor_c05)
        tor_border = load_border(TOR_CFG, dst_crs=tor_plot_crs)
        tor_xl, tor_yl = get_data_extent(tor_mh, tor_pipe, tor_wwtp, pad=0.05)
        if TOR_CFG["xlim"] is not None:
            tor_xl = TOR_CFG["xlim"]
        if TOR_CFG["ylim"] is not None:
            tor_yl = TOR_CFG["ylim"]
        tor_scalebar_m = auto_scalebar_length_m(tor_xl)

        print("=" * 60)
        print("Loading Los Angeles …")
        la_rp, la_c006, la_c05 = load_csvs(LA_CFG)
        la_mh, la_pipe, la_wwtp, la_src_crs, la_plot_crs = auto_prepare_city_spatial_data(
            LA_CFG, la_rp, la_c006, la_c05)
        la_border = load_border(LA_CFG, dst_crs=la_plot_crs)
        la_xl, la_yl = get_data_extent(la_mh, la_pipe, la_wwtp, pad=0.05)
        if LA_CFG["xlim"] is not None:
            la_xl = LA_CFG["xlim"]
        if LA_CFG["ylim"] is not None:
            la_yl = LA_CFG["ylim"]
        la_scalebar_m = auto_scalebar_length_m(la_xl)

        print("=" * 60)
        print("Extracting pipe parameters …")
        all_city_params = {}
        for cname, rp_df in [("Hong Kong", hk_rp), ("Toronto", tor_rp),
                             ("Los Angeles", la_rp)]:
            print(f"  [{cname}]")
            all_city_params[cname] = extract_pipe_params(rp_df, cname)

        print("=" * 60)
        print("Building WWTP predicted-vs-measured comparison …")
        wwtp_res = build_comparison()
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(wwtp_res.to_string(index=False))

        export_figure2_data(
            FIG2_DATA_DIR,
            all_city_params=all_city_params,
            sci_whisker_ranges=sci_whisker_ranges,
            city_mh={"Hong Kong": hk_mh, "Toronto": tor_mh, "Los Angeles": la_mh},
            city_pipes={"Hong Kong": hk_pipe, "Toronto": tor_pipe,
                        "Los Angeles": la_pipe},
            wwtp_res=wwtp_res,
            segment_counts={"Hong Kong": len(hk_pipe),
                            "Toronto": len(tor_pipe),
                            "Los Angeles": len(la_pipe)},
        )
    else:
        print("=" * 60)
        print("LAYOUT_PREVIEW_ONLY = True → skipping data loading, drawing frames only.")

    # ───────────────────────────────────────────────────────────────────────────
    # FIGURE CONSTRUCTION
    # ───────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Creating merged Figure 2 …")

    fig1 = plt.figure(figsize=(18 / 1.6, 22.5 / 1.6), facecolor=BG_FIG)

    LEFT = 0.07
    RIGHT = 0.97
    GAP_X = 0.05
    COL_W = (RIGHT - LEFT - 2 * GAP_X) / 3.0
    col_x = []
    for i in range(3):
        x0 = LEFT + i * (COL_W + GAP_X)
        col_x.append((x0, x0 + COL_W))

    AF_LEFT = col_x[1][0]
    AF_RIGHT = col_x[2][1]
    GAP_AF = 0.045
    AF_PANEL_W = (AF_RIGHT - AF_LEFT - 2 * GAP_AF) / 3.0
    af_x = []
    for i in range(3):
        x0 = AF_LEFT + i * (AF_PANEL_W + GAP_AF)
        af_x.append((x0, x0 + AF_PANEL_W))

    TOP = 0.975
    BOTTOM = 0.04
    GAP_TOP = 0.03
    GAP_MAP = 0.015

    ratios = {
        "top": (2.2 + 2.2) * (2.0 / 3.0),
        "hk": 4.77,
        "tor": 3.4,
        "la": 4.67,
    }
    total_gap = GAP_TOP + 2 * GAP_MAP
    avail = (TOP - BOTTOM) - total_gap
    total_ratio = sum(ratios.values())
    unit = avail / total_ratio

    h_top = ratios["top"] * unit
    h_hk = ratios["hk"] * unit
    h_tor = ratios["tor"] * unit
    h_la = ratios["la"] * unit

    top_top = TOP
    top_bot = TOP - h_top
    hk_top = top_bot - GAP_TOP
    hk_bot = hk_top - h_hk
    tor_top = hk_bot - GAP_MAP
    tor_bot = tor_top - h_tor
    la_top = tor_bot - GAP_MAP
    la_bot = la_top - h_la

    GAP_AF_V = 0.030
    af_row_h = (h_top - GAP_AF_V) / 2.0
    row0_top = top_top
    row0_bot = top_top - af_row_h
    row1_top = row0_bot - GAP_AF_V
    row1_bot = top_bot

    param_axes = [
        _ax_rect(fig1, af_x[0], (row0_bot, row0_top)),
        _ax_rect(fig1, af_x[1], (row0_bot, row0_top)),
        _ax_rect(fig1, af_x[2], (row0_bot, row0_top)),
        _ax_rect(fig1, af_x[0], (row1_bot, row1_top)),
        _ax_rect(fig1, af_x[1], (row1_bot, row1_top)),
        _ax_rect(fig1, af_x[2], (row1_bot, row1_top)),
    ]

    ax_b = _ax_rect(fig1, col_x[0], (top_bot, top_top))

    r_hk = [_ax_rect(fig1, col_x[j], (hk_bot, hk_top)) for j in range(3)]
    r_tor = [_ax_rect(fig1, col_x[j], (tor_bot, tor_top)) for j in range(3)]
    r_la = [_ax_rect(fig1, col_x[j], (la_bot, la_top)) for j in range(3)]

    if LAYOUT_PREVIEW_ONLY:
        for ax, lbl in zip(param_axes, PARAM_LABELS):
            _layout_placeholder(ax, lbl)
        _layout_placeholder(ax_b, "a", "Predicted vs measured WWTP inflow")
        for ax, lbl, t in zip(r_hk, ["c", "d", "e"],
                              ["HK – H2S", "HK – dispersion", "HK – corrosion"]):
            _layout_placeholder(ax, lbl, t)
        for ax, lbl, t in zip(r_tor, ["f", "g", "h"],
                              ["TOR – H2S", "TOR – dispersion", "TOR – corrosion"]):
            _layout_placeholder(ax, lbl, t)
        for ax, lbl, t in zip(r_la, ["i", "j", "k"],
                              ["LA – H2S", "LA – dispersion", "LA – corrosion"]):
            _layout_placeholder(ax, lbl, t)
    else:
        plot_param_panels(
            param_axes, all_city_params, PARAM_ORDER, PARAM_UNITS,
            CITY_ORDER, CITY_COLORS, PARAM_LABELS,
            sci_whisker_ranges=sci_whisker_ranges,
            strip_sample_ratio=0.10, min_strip_pts=50
        )

        plot_wwtp_comparison(ax_b, wwtp_res, label="a", use_log=PANEL_B_USE_LOG)

        plot_map_row(
            r_hk, hk_mh, hk_pipe, wwtp=hk_wwtp, maint_gdf=hk_maint,
            border_gdf=hk_border, xlim=hk_xl, ylim=hk_yl,
            labels=("c", "d", "e"), city_name="Hong Kong",
            plot_crs=hk_plot_crs, is_hk=True, scalebar_m=HK_CFG["scalebar_m"],
            show_legend=True, disp_radius_scale=DISP_RADIUS_SCALE,
            disp_outlier_percent=DISP_RADIUS_OUTLIER_PERCENT
        )
        plot_map_row(
            r_tor, tor_mh, tor_pipe, wwtp=tor_wwtp, maint_gdf=None,
            border_gdf=tor_border, xlim=tor_xl, ylim=tor_yl,
            labels=("f", "g", "h"), city_name="Toronto",
            plot_crs=tor_plot_crs, is_hk=False, scalebar_m=tor_scalebar_m,
            show_legend=False, disp_radius_scale=DISP_RADIUS_SCALE,
            disp_outlier_percent=DISP_RADIUS_OUTLIER_PERCENT
        )
        plot_map_row(
            r_la, la_mh, la_pipe, wwtp=la_wwtp, maint_gdf=None,
            border_gdf=la_border, xlim=la_xl, ylim=la_yl,
            labels=("i", "j", "k"), city_name="Los Angeles",
            plot_crs=la_plot_crs, is_hk=False, scalebar_m=la_scalebar_m,
            show_legend=False, disp_radius_scale=DISP_RADIUS_SCALE,
            disp_outlier_percent=DISP_RADIUS_OUTLIER_PERCENT
        )

    leg_handles = make_param_legend_handles(CITY_ORDER, CITY_COLORS)
    legend_cx = (af_x[0][0] + af_x[2][1]) / 2.0
    legend_y = row1_bot - 0.012
    leg_a = fig1.legend(
        handles=leg_handles, loc="upper center",
        bbox_to_anchor=(legend_cx, legend_y),
        ncol=len(leg_handles), fontsize=5.4, frameon=True,
        framealpha=0.95, edgecolor="none",
        columnspacing=1.0, handletextpad=0.4, borderpad=0.4
    )
    leg_a.get_frame().set_facecolor("white")

    out_png = os.path.join(OUTPUT_DIR, f"{TAG}.png")
    fig1.savefig(out_png, dpi=600, bbox_inches="tight",
                 facecolor=BG_FIG, transparent=False)

    if not args.no_show:
        plt.show()

    print("\n✓ Merged Figure 2 generated and data exported.")
    print(f"✓ PANEL_B_USE_LOG = {PANEL_B_USE_LOG} ; LAYOUT_PREVIEW_ONLY = {LAYOUT_PREVIEW_ONLY}")
    print(f"✓ DISP_RADIUS_SCALE = {DISP_RADIUS_SCALE}")
    print(f"✓ Saved 600 dpi PNG to:\n  {out_png}")
    print(f"✓ Data for results calculator:\n  {FIG2_DATA_DIR}")
