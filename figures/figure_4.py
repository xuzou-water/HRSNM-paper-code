# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:54:18 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 21:12:29 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
Figure 4 — Nature-style mechanistic summary
Refactored to consume Code-1 outputs (v7) for HK / Toronto / LA.
Outfalls are identified from hydraulic_nodes.csv (type=='outfall'),
so the .inp file is NOT needed anymore.
"""

import os
import time
import json
import gc
import warnings
from collections import deque
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from scipy import stats
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist
from shapely.geometry import Point
from shapely.ops import unary_union

import networkx as nx

from hrsnm.corrosion import (
    DEFAULT_CORROSION_RATE_THRESHOLD_MM_PER_YEAR,
    corrosion_exceedance_length_km,
)
from hrsnm.scenarios import scenario_axes, scenario_count

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


# ================================================================
# 0. RUN MODE 12h for N_Workers = 1
# ================================================================
RUN_PREPROCESSING = os.environ.get("HRSNM_FIG4_FROM_CACHE", "0") != "1"
SAVE_CACHE = True
N_WORKERS = int(os.environ.get("HRSNM_FIG4_WORKERS", "1"))
NO_SHOW = os.environ.get("HRSNM_FIG4_NO_SHOW", "0") == "1"
EXPECTED_CATCHMENTS = int(os.environ.get("HRSNM_FIG4_EXPECTED_CATCHMENTS", "17"))

SCENARIO_GRID = os.environ.get("HRSNM_FIG4_SCENARIO_GRID", "small8")
CACHE_TAG = (
    f"figure4_v7_hk_to_la_cityprefixed_{SCENARIO_GRID}_corr_gt_1mm_per_year"
)

# small8 is the local 2×2×2 corner grid; full27 is the manuscript/server grid.
EXCLUDE_TEMPERATURES = set()
EXCLUDE_SO4 = set()
N_SCENARIOS_KEPT = scenario_count(SCENARIO_GRID)


# ================================================================
# 1. GLOBAL STYLE
# ================================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

TEXT_C = "#1A1A1A"
SUBTEXT_C = "#666666"
SPINE_C = "#222222"
GRID_C = "#D8D8D8"

CITY_COLORS = {
    "Hong Kong": "#3A7A8C",
    "Toronto": "#A66842",
    "Los Angeles": "#A1B4AC",
}
CITY_MARKERS = {
    "Hong Kong": "D",
    "Toronto": "o",
    "Los Angeles": "s",
}
RISK_COLORS = {
    "Problem_Length_km": "#08345A",
    "Toxic_Length_km": "#9C3106",
    "Contaminated_Area_km2": "#3A7A8C",
}

EPS = 1e-10
BUILD_Q_MIN_M3_PER_D = float(os.environ.get("HRSNM_FIG4_BUILD_Q_MIN_M3_D", "0.001"))
BUILD_Q_MIN_TAG = (
    f"qb_gt_{str(BUILD_Q_MIN_M3_PER_D).replace('.', 'p')}_m3d"
    if BUILD_Q_MIN_M3_PER_D > 0
    else "qb_gt_0_m3d"
)
SH2S_COL = "SH2S"
H2S_TOXIC_THRESHOLD_PPM = 10
SECONDS_PER_DAY = 24.0 * 3600.0
DAYS_PER_YEAR = 365.0
CORROSION_RATE_THRESHOLD_MM_PER_YEAR = (
    DEFAULT_CORROSION_RATE_THRESHOLD_MM_PER_YEAR
)


# ================================================================
# 2. PATHS & SCENARIOS  (Code-1 v7 outputs)
# ================================================================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_BASE = os.environ.get(
    "HRSNM_FIG4_DATA_BASE",
    os.path.join(REPO_ROOT, "data", "processed_data"),
)

output_dir = os.path.abspath(os.environ.get(
    "HRSNM_FIG4_OUT_DIR",
    os.path.join(
        REPO_ROOT,
        "fig4_results_export",
    ),
))

# --- 三个城市的数据布局 ----------------------------------------
# 假定 HK / LA 与 Toronto 一致：
#   {DATA_BASE}/{city_dir}/biochemical_results/{prefix}_result_pipes_v7_{TAG}.csv
#   {DATA_BASE}/{city_dir}/biochemical_results/{prefix}_concentration_results_*_v7_{TAG}.csv
#   {DATA_BASE}/{city_dir}/hydraulic_results/{prefix}_building_details.csv
#   {DATA_BASE}/{city_dir}/hydraulic_nodes.csv
CITY_CONFIG = {
    "Hong Kong": {
        "data_dir": os.path.join(DATA_BASE, "hk_v3"),
        "biochem_dir": os.environ.get("HRSNM_FIG4_HK_BIOCHEM_DIR"),
        "prefix": "hk",
        "abbrev_prefix": "HK",
    },
    "Toronto": {
        "data_dir": os.path.join(DATA_BASE, "toronto_v3"),
        "biochem_dir": os.environ.get("HRSNM_FIG4_TORONTO_BIOCHEM_DIR"),
        "prefix": "toronto",
        "abbrev_prefix": "TO",
    },
    "Los Angeles": {
        "data_dir": os.path.join(DATA_BASE, "la_v3"),
        "biochem_dir": os.environ.get("HRSNM_FIG4_LA_BIOCHEM_DIR"),
        "prefix": "la",
        "abbrev_prefix": "LA",
    },
}
EXPECTED_OUTFALLS_BY_CITY = {
    "Hong Kong": 9,
    "Toronto": 5,
    "Los Angeles": 3,
}

# Scenario space is shared with the three city HRSNM scripts.
TEMPERATURES, SO4S, CODS = scenario_axes(SCENARIO_GRID)

SCENARIOS = [
    {
        "TAG": f"T_{T}_SO4_{SO4}_COD_{COD}",
        "Temperature": T,
        "SO4": SO4,
        "COD": COD,
    }
    for T in TEMPERATURES
    for SO4 in SO4S
    for COD in CODS
]
EXPECTED_SCENARIO_TAGS = [s["TAG"] for s in SCENARIOS]
print(f"[Scenarios] total = {len(SCENARIOS)}")

CACHE_AGGREGATED_PATH = os.path.join(
    output_dir, f"cached_aggregated_{CACHE_TAG}.csv"
)
CACHE_RESULTS_PATH = os.path.join(
    output_dir, f"cached_results_{CACHE_TAG}.csv"
)
CACHE_ABBREV_PATH = os.path.join(
    output_dir, f"cached_abbrev_map_{CACHE_TAG}.json"
)


# ================================================================
# 3. LABELS
# ================================================================
SHORT_PARAM = {
    "average_slope": "Slope",
    "average_velocity": "Velocity",
    "average_HRT": "HRT",
    "total_length_km": "Sewer length",
    "average_AV": "A/V",
    "flowrate_to_STW": r"$Q_{\mathrm{WWTP}}$",
    "average_SO4": "Sulphate",
    "average_sulphide": "TDS",
    "average_methane": r"CH$_4$",
    "average_do": "DO",
    "sum_L_build": r"$\sum L_b$",
    "ave_L_build": r"Mean $L_b$",
    "sum_L_build_Q": r"$\sum (L_b Q_b)$",
    "sum_L_build_over_Q": r"$\sum (L_b/Q_b)$",
    "sum_Q_build": r"$\sum Q_b$",
    "sum_L_build_over_sum_Q_build": r"$\sum L_b/\sum Q_b$",
    "total_flow": "Network flow",
}

UNIT_PARAM = {
    "average_slope": "-",
    "average_velocity": r"m s$^{-1}$",
    "average_HRT": "h",
    "total_length_km": "km",
    "average_AV": r"m$^{-1}$",
    "flowrate_to_STW": r"m$^3$ d$^{-1}$",
    "average_SO4": r"mg L$^{-1}$",
    "average_sulphide": r"mg L$^{-1}$",
    "average_methane": r"mg L$^{-1}$",
    "average_do": r"mg L$^{-1}$",
    "sum_L_build": "m",
    "ave_L_build": "m",
    "sum_L_build_Q": r"m$^4$ d$^{-1}$",
    "sum_L_build_over_Q": r"m d m$^{-3}$",
    "sum_Q_build": r"m$^3$ d$^{-1}$",
    "sum_L_build_over_sum_Q_build": r"m d m$^{-3}$",
    "total_flow": r"m$^3$ d$^{-1}$",
}

RISKS = ["Problem_Length_km", "Toxic_Length_km", "Contaminated_Area_km2"]

RISK_SHORT = {
    "Problem_Length_km": "Corrosion",
    "Toxic_Length_km": "Toxicity",
    "Contaminated_Area_km2": "Odour",
}
RISK_LABEL = {
    "Problem_Length_km": r"Corrosion exceedance length, $L_{\mathrm{corr}}$",
    "Toxic_Length_km": r"Toxic length, $L_{\mathrm{tox}}$",
    "Contaminated_Area_km2": r"Odour-affected area, $A_{\mathrm{odor}}$",
}
RISK_UNIT = {
    "Problem_Length_km": "km",
    "Toxic_Length_km": "km",
    "Contaminated_Area_km2": r"km$^2$",
}


# ================================================================
# 4. HELPERS
# ================================================================
def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_csv_checked(path, usecols=None):
    if not os.path.exists(path):
        print(f"[Missing file] {path}")
        return None
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception as e:
        print(f"[Read error] {path}: {e}")
        return None


def batched_unary_union(geoms, batch_size=2000):
    geoms = list(geoms)
    if not geoms:
        return None
    if len(geoms) <= batch_size:
        return unary_union(geoms)
    partials = []
    for i in range(0, len(geoms), batch_size):
        partials.append(unary_union(geoms[i:i + batch_size]))
        gc.collect()
    out = unary_union(partials)
    del partials
    gc.collect()
    return out


def trace_to_terminal(node, G, targets):
    if node in targets:
        return node
    visited = set()
    queue = deque([node])
    while queue:
        n = queue.popleft()
        if n in visited:
            continue
        visited.add(n)
        for s in G.successors(n):
            if s in targets:
                return s
            if s not in visited:
                queue.append(s)
    return None


def add_scenario_cols(df, tag, T, SO4, COD):
    df = df.copy()
    df["Scenario"] = tag
    df["Temperature"] = T
    df["SO4"] = SO4
    df["COD"] = COD
    return df


def p_to_stars(p):
    if p is None or np.isnan(p):
        return "n.s."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def format_p_value(p):
    if p is None or np.isnan(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def format_power_law_equation(a, b):
    """Return a Nature-style power law with a true superscript base-10 exponent."""
    if a is None or not np.isfinite(a) or a == 0:
        coefficient = "0"
    else:
        exponent = int(np.floor(np.log10(abs(a))))
        mantissa = a / (10.0 ** exponent)
        sign = "-" if mantissa < 0 else ""
        coefficient = (
            rf"{sign}{abs(mantissa):.2g}\times10^{{{exponent:d}}}"
        )
    return rf"$y={coefficient}x^{{{b:.2f}}}$"


def format_p_annotation(p):
    """Format model P values without the invalid '= <' construction."""
    if p is None or np.isnan(p):
        return r"$P=\mathrm{NA}$"
    if p < 0.001:
        return r"$P<0.001$"
    return rf"$P={p:.3f}$"


def apply_axis_style(ax, half_frame=True):
    ax.set_facecolor("white")
    if half_frame:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_visible(True)
            ax.spines[s].set_color(SPINE_C)
            ax.spines[s].set_linewidth(0.7)
    else:
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(SPINE_C)
            s.set_linewidth(0.7)
    ax.tick_params(axis="both", colors=TEXT_C, labelsize=7,
                   direction="out", width=0.6, length=2.5, pad=2)


def panel_label(ax, label, x=-0.20, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", ha="left", va="bottom", color=TEXT_C)


def city_prefix_for_nodes(city_label):
    return {"Toronto": "Toronto",
            "Los Angeles": "LosAngeles",
            "Hong Kong": "HongKong"}.get(city_label, str(city_label).replace(" ", ""))


def prefix_wwtp_name(value, city_label):
    if pd.isna(value):
        return value
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return s
    prefix = city_prefix_for_nodes(city_label)
    if s.startswith(prefix + "_"):
        return s
    if s.upper().startswith("WWTP"):
        return f"{prefix}_{s}"
    return s


def prefix_city_wwtp_columns(df, city_label, columns):
    if df is None:
        return None
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: prefix_wwtp_name(x, city_label))
    return df


def ensure_q_units_m3_per_day(df):
    """Return a copy with every volumetric-flow quantity expressed per day.

    Legacy Figure 4 caches explicitly marked ``m3/year`` are migrated using
    the 365-day factor with which they were created. Missing or mixed unit
    metadata is rejected so that cached values cannot be converted twice.
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    if "Q_UNIT" not in df.columns:
        raise ValueError("Q_UNIT metadata is required for Figure 4 flow data.")
    units = set(df["Q_UNIT"].dropna().astype(str).str.strip())
    if units == {"m3/day"}:
        return df
    if units != {"m3/year"}:
        raise ValueError(f"Unsupported or mixed Figure 4 Q_UNIT values: {sorted(units)}")

    flow_like = [
        "flowrate_to_STW", "total_flow", "sum_L_build_Q", "sum_Q_build",
    ]
    inverse_flow_like = [
        "sum_L_build_over_Q", "sum_L_build_over_sum_Q_build",
    ]
    for c in flow_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") / DAYS_PER_YEAR
    for c in inverse_flow_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") * DAYS_PER_YEAR
    df["Q_UNIT"] = "m3/day"
    return df


def add_building_sum_ratio_columns(df):
    """Backfill building-flow metrics and thresholded sum(Lb/Qb) for caches."""
    if df is None or df.empty:
        return df
    if not {"City", "Assigned_Outfall"}.issubset(df.columns):
        return df

    rows = []
    summary_rows = []
    for city_label, config in CITY_CONFIG.items():
        hyd_dir = os.path.join(config["data_dir"], "hydraulic_results")
        bldg_path = os.path.join(hyd_dir, f"{config['prefix']}_building_details.csv")
        usecols = ["endnode_wwtp", "dist_to_wwtp_m", "wastewater_flow_m3_per_d"]
        c_bldg = read_csv_checked(bldg_path, usecols=usecols)
        if c_bldg is None or "endnode_wwtp" not in c_bldg.columns:
            continue
        c_bldg = prefix_city_wwtp_columns(c_bldg, city_label, ["endnode_wwtp"])
        safe_numeric(c_bldg, ["dist_to_wwtp_m", "wastewater_flow_m3_per_d"])
        c_bldg["Q_m3d"] = c_bldg["wastewater_flow_m3_per_d"]
        c_bldg["L_Q"] = c_bldg["dist_to_wwtp_m"] * c_bldg["Q_m3d"]
        valid_q = (
            c_bldg["wastewater_flow_m3_per_d"].notna()
            & (c_bldg["wastewater_flow_m3_per_d"] > 0)
            & c_bldg["dist_to_wwtp_m"].notna()
        )
        keep_l_over_q = (
            valid_q
            & (c_bldg["wastewater_flow_m3_per_d"] > BUILD_Q_MIN_M3_PER_D)
        )
        c_bldg["L_over_Q"] = 0.0
        c_bldg.loc[keep_l_over_q, "L_over_Q"] = (
            c_bldg.loc[keep_l_over_q, "dist_to_wwtp_m"]
            / c_bldg.loc[keep_l_over_q, "Q_m3d"]
        )
        c_bldg["building_count"] = (
            c_bldg["dist_to_wwtp_m"].notna()
            & c_bldg["Q_m3d"].notna()
        ).astype(int)
        c_bldg["building_count_l_over_q"] = keep_l_over_q.astype(int)
        raw_l_over_q = (
            c_bldg.loc[valid_q, "dist_to_wwtp_m"]
            / c_bldg.loc[valid_q, "Q_m3d"]
        )
        raw_sum = float(raw_l_over_q.sum())
        kept_sum = float(c_bldg.loc[keep_l_over_q, "L_over_Q"].sum())
        summary_rows.append({
            "City": city_label,
            "valid_buildings": int(valid_q.sum()),
            "excluded_buildings": int((valid_q & ~keep_l_over_q).sum()),
            "excluded_buildings_percent": (
                100.0 * int((valid_q & ~keep_l_over_q).sum()) / int(valid_q.sum())
                if int(valid_q.sum()) else np.nan
            ),
            "removed_sum_L_over_Q_percent": (
                100.0 * (raw_sum - kept_sum) / raw_sum if raw_sum > 0 else np.nan
            ),
        })
        city_rows = (
            c_bldg.groupby("endnode_wwtp").agg(
                sum_L_build=("dist_to_wwtp_m", "sum"),
                sum_Q_build=("Q_m3d", "sum"),
                sum_L_build_Q=("L_Q", "sum"),
                sum_L_build_over_Q=("L_over_Q", "sum"),
                n_buildings=("building_count", "sum"),
                n_buildings_l_over_q=("building_count_l_over_q", "sum"),
            ).reset_index().rename(columns={"endnode_wwtp": "Assigned_Outfall"})
        )
        city_rows["sum_L_build_over_sum_Q_build"] = (
            city_rows["sum_L_build"]
            / city_rows["sum_Q_build"].where(city_rows["sum_Q_build"] > EPS)
        )
        city_rows["City"] = city_label
        rows.append(city_rows)

    out = df.copy()
    if rows:
        lookup = pd.concat(rows, ignore_index=True)
        out = out.merge(
            lookup[[
                "City", "Assigned_Outfall", "sum_L_build", "sum_Q_build",
                "sum_L_build_Q", "sum_L_build_over_Q",
                "sum_L_build_over_sum_Q_build",
                "n_buildings", "n_buildings_l_over_q",
            ]],
            on=["City", "Assigned_Outfall"],
            how="left",
            suffixes=("", "_from_buildings"),
        )
        for col in [
            "sum_L_build", "sum_Q_build", "sum_L_build_Q",
            "sum_L_build_over_Q", "sum_L_build_over_sum_Q_build",
            "n_buildings", "n_buildings_l_over_q",
        ]:
            from_col = f"{col}_from_buildings"
            if from_col in out.columns:
                out[col] = out[from_col]
                out = out.drop(columns=[from_col])
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(
            "[Building Q threshold] "
            f"Q_b > {BUILD_Q_MIN_M3_PER_D:g} m3/d for sum(Lb/Qb):"
        )
        print(summary_df.to_string(index=False))
    return out


def filter_scenarios_keep(df, exclude_T=EXCLUDE_TEMPERATURES, exclude_SO4=EXCLUDE_SO4):
    if df is None or df.empty:
        return df
    if not exclude_T and not exclude_SO4:
        return df.reset_index(drop=True)
    df = df.copy()
    n0 = len(df)
    if "Temperature" in df.columns and exclude_T:
        df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
        df = df[~df["Temperature"].isin(exclude_T)]
    if "SO4" in df.columns and exclude_SO4:
        df["SO4"] = pd.to_numeric(df["SO4"], errors="coerce")
        df = df[~df["SO4"].isin(exclude_SO4)]
    print(f"[filter] rows: {n0} -> {len(df)}")
    return df.reset_index(drop=True)


# ================================================================
# 5. SINGLE CITY PROCESSING (unified for HK / Toronto / LA)
# ================================================================
def process_one_city_scenario(city_label, config, TAG, T, SO4, COD):
    """Returns (results_df, agg_df, abbrev_map) for one city."""
    data_dir = config["data_dir"]
    prefix = config["prefix"]
    ab_pre = config["abbrev_prefix"]

    biochem_dir = config.get("biochem_dir") or os.path.join(
        data_dir, "biochemical_results"
    )
    hyd_dir = os.path.join(data_dir, "hydraulic_results")

    pipes_path = os.path.join(biochem_dir, f"{prefix}_result_segments_v7_{TAG}.csv")
    em_path = os.path.join(
        biochem_dir,
        f"{prefix}_emission_radius_segments_v7_{TAG}.csv"
    )
    em_od_path = os.path.join(
        biochem_dir,
        f"{prefix}_emission_radius_odour_segments_v7_{TAG}.csv"
    )
    bldg_path = os.path.join(hyd_dir, f"{prefix}_building_details.csv")
    nodes_path = os.path.join(data_dir, "hydraulic_nodes.csv")

    c_res = read_csv_checked(pipes_path)
    c_em = read_csv_checked(em_path)
    c_em_od = read_csv_checked(em_od_path)
    c_bldg = read_csv_checked(bldg_path)
    c_nodes = read_csv_checked(nodes_path)

    if any(x is None for x in [c_res, c_em, c_em_od, c_bldg, c_nodes]):
        print(f"[Skipped] {city_label} / {TAG}: missing input file(s).")
        return None

    # ---- 列重命名：与代码1一致 ----
    if "v" in c_res.columns and "velocity" not in c_res.columns:
        c_res = c_res.rename(columns={"v": "velocity"})

    # ---- 城市内 WWTP 名加前缀 ----
    c_res = prefix_city_wwtp_columns(c_res, city_label, ["start", "end"])
    c_em = prefix_city_wwtp_columns(c_em, city_label, ["node_name"])
    c_em_od = prefix_city_wwtp_columns(c_em_od, city_label, ["node_name"])
    c_bldg = prefix_city_wwtp_columns(c_bldg, city_label, ["endnode_wwtp"])
    if "node" in c_nodes.columns:
        c_nodes["node"] = c_nodes["node"].astype(str).apply(
            lambda x: prefix_wwtp_name(x, city_label)
        )

    # ---- 识别出水口：替代 INP 文件 ----
    if "type" in c_nodes.columns:
        outfall_nodes_from_inp = set(
            c_nodes.loc[c_nodes["type"].astype(str).str.lower() == "outfall", "node"]
            .astype(str).tolist()
        )
    else:
        outfall_nodes_from_inp = set()

    # 建有向图
    cG = nx.DiGraph()
    for _, row in c_res.iterrows():
        s = str(row.get("start", ""))
        e = str(row.get("end", ""))
        if s and e and s != "nan" and e != "nan":
            cG.add_edge(s, e)

    gn = set(cG.nodes())
    target_set = outfall_nodes_from_inp.intersection(gn)
    missing_declared_outfalls = sorted(outfall_nodes_from_inp - gn)
    if missing_declared_outfalls:
        raise ValueError(
            f"{city_label} / {TAG}: declared outfalls absent from the scenario "
            f"network: {missing_declared_outfalls}"
        )
    if not target_set:
        target_set = {n for n in cG.nodes() if cG.out_degree(n) == 0}
    expected_outfalls = EXPECTED_OUTFALLS_BY_CITY[city_label]
    if len(target_set) != expected_outfalls:
        raise ValueError(
            f"{city_label} / {TAG}: expected {expected_outfalls} outfalls, "
            f"found {len(target_set)}: {sorted(target_set)}"
        )

    # 节点 -> 终端出水口
    node_w = {n: trace_to_terminal(n, cG, target_set) for n in cG.nodes()}
    c_res["Assigned_WWTP"] = c_res["start"].astype(str).map(node_w)
    c_res = c_res.dropna(subset=["Assigned_WWTP"]).copy()
    c_em["Assigned_WWTP"] = c_em["node_name"].astype(str).map(node_w)
    c_em_od["Assigned_WWTP"] = c_em_od["node_name"].astype(str).map(node_w)

    # 终端缩写映射
    ab_map = {n: f"{ab_pre}-{i + 1}" for i, n in enumerate(sorted(target_set))}

    # ---- 聚合管段属性（按 catchment / outfall） ----
    c_grav = c_res.copy()
    safe_numeric(c_grav, [
        "slope", "velocity", "HRT", "length", "A_V",
        "SHS_in", "CH4_in", "SO_in", "SSO4_in", "flowrate",
    ])

    c_agg = (
        c_grav.groupby("Assigned_WWTP").agg(
            average_slope=("slope", "mean"),
            average_velocity=("velocity", "mean"),
            average_HRT=("HRT", "mean"),
            total_length=("length", "sum"),
            average_AV=("A_V", "mean"),
            average_sulphide=("SHS_in", "mean"),
            average_methane=("CH4_in", "mean"),
            average_do=("SO_in", "mean"),
            average_SO4=("SSO4_in", "mean"),
        ).reset_index().rename(columns={"Assigned_WWTP": "Assigned_Outfall"})
    )

    # 进入 STW 的流量：在出水口处求和
    c_end = c_res[c_res["end"].astype(str).isin(target_set)].copy()
    safe_numeric(c_end, ["flowrate"])
    f_dict = c_end.groupby(c_end["end"].astype(str))["flowrate"].sum().to_dict()
    c_agg["flowrate_to_STW"] = c_agg["Assigned_Outfall"].map(f_dict)
    c_agg["flowrate_to_STW"] = (
        pd.to_numeric(c_agg["flowrate_to_STW"], errors="coerce") * SECONDS_PER_DAY
    )
    c_agg["total_length_km"] = c_agg["total_length"] / 1000.0

    # ---- 建筑（来自 hydraulic_results/{prefix}_building_details.csv） ----
    safe_numeric(c_bldg, ["wastewater_flow_m3_per_d", "dist_to_wwtp_m"])
    if "endnode_wwtp" not in c_bldg.columns:
        cb_agg = pd.DataFrame(columns=[
            "Assigned_Outfall", "sum_L_build", "ave_L_build",
            "sum_L_build_Q", "sum_L_build_over_Q", "sum_Q_build",
            "sum_L_build_over_sum_Q_build",
            "n_buildings", "n_buildings_l_over_q",
        ])
    else:
        c_bldg["Q_m3d"] = c_bldg["wastewater_flow_m3_per_d"]
        c_bldg["L_Q"] = c_bldg["dist_to_wwtp_m"] * c_bldg["Q_m3d"]
        mq = (
            (c_bldg["Q_m3d"] > EPS)
            & (c_bldg["wastewater_flow_m3_per_d"] > BUILD_Q_MIN_M3_PER_D)
        )
        c_bldg["L_over_Q"] = 0.0
        c_bldg.loc[mq, "L_over_Q"] = (
            c_bldg.loc[mq, "dist_to_wwtp_m"] / c_bldg.loc[mq, "Q_m3d"]
        )
        c_bldg["building_count"] = (
            c_bldg["dist_to_wwtp_m"].notna()
            & c_bldg["Q_m3d"].notna()
        ).astype(int)
        c_bldg["building_count_l_over_q"] = mq.astype(int)
        cb_agg = (
            c_bldg.groupby("endnode_wwtp").agg(
                sum_L_build=("dist_to_wwtp_m", "sum"),
                ave_L_build=("dist_to_wwtp_m", "mean"),
                sum_Q_build=("Q_m3d", "sum"),
                sum_L_build_Q=("L_Q", "sum"),
                sum_L_build_over_Q=("L_over_Q", "sum"),
                n_buildings=("building_count", "sum"),
                n_buildings_l_over_q=("building_count_l_over_q", "sum"),
            ).reset_index().rename(columns={"endnode_wwtp": "Assigned_Outfall"})
        )
        cb_agg["sum_L_build_over_sum_Q_build"] = (
            cb_agg["sum_L_build"]
            / cb_agg["sum_Q_build"].where(cb_agg["sum_Q_build"] > EPS)
        )

    c_agg = pd.merge(c_agg, cb_agg, on="Assigned_Outfall", how="left")

    # ---- 嗅觉污染面积 ----
    c_em_od2 = c_em_od.dropna(subset=["Assigned_WWTP"]).copy()
    safe_numeric(c_em_od2, ["distance", "us_x", "us_y"])
    c_em_od2 = c_em_od2.dropna(subset=["distance", "us_x", "us_y"])
    c_em_od2 = c_em_od2[
        (c_em_od2["distance"] > 0)
        & np.isfinite(c_em_od2[["distance", "us_x", "us_y"]]).all(axis=1)
    ]

    a_recs = []
    for w, grp in c_em_od2.groupby("Assigned_WWTP"):
        circles = (
            Point(r["us_x"], r["us_y"]).buffer(r["distance"], resolution=8)
            for _, r in grp.iterrows()
        )
        geom = batched_unary_union(circles, batch_size=1500)
        a_recs.append({
            "Assigned_Outfall": w,
            "Contaminated_Area_km2": 0.0 if geom is None else geom.area / 1e6,
        })
        del geom
        gc.collect()
    c_area_df = pd.DataFrame(a_recs)

    # ---- 腐蚀/有毒长度 ----
    c_em2 = c_em.dropna(subset=["Assigned_WWTP"]).copy()
    safe_numeric(c_em2, ["dcorr_dt", "length", SH2S_COL])
    # Corrosion exceedance is defined directly by annual rate; structural and
    # hydraulic dimensions are not part of this classification.
    c_ms = c_em2
    c_ms["is_toxic"] = c_ms[SH2S_COL] > H2S_TOXIC_THRESHOLD_PPM

    c_recs = []
    for w, ab in ab_map.items():
        cd = c_ms[c_ms["Assigned_WWTP"] == w]
        pl = corrosion_exceedance_length_km(
            cd["dcorr_dt"].to_numpy(),
            cd["length"].to_numpy(),
            threshold_mm_per_year=CORROSION_RATE_THRESHOLD_MM_PER_YEAR,
        )
        tl = cd.loc[cd["is_toxic"], "length"].sum() / 1000.0
        c_recs.append({
            "Catchment": w, "Abbrev": ab,
            "Problem_Length_km": pl, "Toxic_Length_km": tl,
            "Maintenance_Cost": pl * 5 * 0.13,
            "Corrosion_Rate_Threshold_mm_per_year": (
                CORROSION_RATE_THRESHOLD_MM_PER_YEAR
            ),
        })
    c_results_df = pd.DataFrame(c_recs)

    # ---- catchment 总流量 ----
    c_total_flow = (
        c_grav.groupby("Assigned_WWTP")["flowrate"].sum().reset_index()
        .rename(columns={"Assigned_WWTP": "Catchment", "flowrate": "total_flow"})
    )
    c_total_flow["total_flow"] = (
        pd.to_numeric(c_total_flow["total_flow"], errors="coerce") * SECONDS_PER_DAY
    )

    c_results_df = pd.merge(c_results_df, c_total_flow, on="Catchment", how="left")
    c_results_df = pd.merge(
        c_results_df,
        c_area_df.rename(columns={"Assigned_Outfall": "Catchment"}),
        on="Catchment", how="left",
    )
    c_results_df = pd.merge(
        c_results_df,
        c_agg[["Assigned_Outfall", "total_length_km", "sum_L_build",
               "sum_L_build_over_Q", "sum_L_build_Q", "sum_Q_build",
               "sum_L_build_over_sum_Q_build"]]
        .rename(columns={"Assigned_Outfall": "Catchment"}),
        on="Catchment", how="left",
    )
    c_agg = pd.merge(
        c_agg,
        c_results_df[["Catchment", "Problem_Length_km", "Maintenance_Cost",
                      "Toxic_Length_km", "Contaminated_Area_km2", "total_flow",
                      "Corrosion_Rate_Threshold_mm_per_year"]]
        .rename(columns={"Catchment": "Assigned_Outfall"}),
        on="Assigned_Outfall", how="left",
    )

    c_results_df["City"] = city_label
    c_agg["City"] = city_label
    c_results_df["Q_UNIT"] = "m3/day"
    c_agg["Q_UNIT"] = "m3/day"

    del c_res, c_em, c_em_od, c_bldg, c_nodes, c_grav, c_end, cb_agg
    del c_em_od2, c_area_df, c_em2, c_ms, c_total_flow
    gc.collect()

    return c_results_df, c_agg, ab_map


def process_one_scenario(TAG, T, SO4, COD):
    print(f"\n[Scenario start] {TAG}")
    all_results, all_agg = [], []
    abbrev_all = {}
    for city_label, config in CITY_CONFIG.items():
        out = process_one_city_scenario(city_label, config, TAG, T, SO4, COD)
        if out is None:
            print(f"[Scenario abort] {TAG}: {city_label} failed.")
            return None
        c_res, c_agg, ab_map = out
        all_results.append(c_res)
        all_agg.append(c_agg)
        abbrev_all.update(ab_map)

    results_df = pd.concat(all_results, ignore_index=True)
    aggregated_df = pd.concat(all_agg, ignore_index=True)
    results_df["Abbrev"] = results_df["Catchment"].map(abbrev_all)
    if "Assigned_Outfall" in aggregated_df.columns:
        aggregated_df["Abbrev"] = aggregated_df["Assigned_Outfall"].map(abbrev_all)

    results_df = add_scenario_cols(results_df, TAG, T, SO4, COD)
    aggregated_df = add_scenario_cols(aggregated_df, TAG, T, SO4, COD)
    results_df["Q_UNIT"] = "m3/day"
    aggregated_df["Q_UNIT"] = "m3/day"

    print(f"[Scenario done]  {TAG}")
    return {
        "TAG": TAG,
        "results_df": results_df,
        "aggregated_df": aggregated_df,
        "all_abbrev_map": abbrev_all,
    }


def _worker(sc):
    try:
        return process_one_scenario(sc["TAG"], sc["Temperature"], sc["SO4"], sc["COD"])
    except Exception as e:
        print(f"[Worker error] {sc['TAG']}: {e}")
        gc.collect()
        return None


def _city_worker(task):
    """Process one city/scenario pair so full27 can use up to 81 workers."""
    city_label = task["City"]
    tag = task["TAG"]
    try:
        out = process_one_city_scenario(
            city_label,
            CITY_CONFIG[city_label],
            tag,
            task["Temperature"],
            task["SO4"],
            task["COD"],
        )
        if out is None:
            return None
        results_df, aggregated_df, abbrev_map = out
        results_df["Abbrev"] = results_df["Catchment"].map(abbrev_map)
        if "Assigned_Outfall" in aggregated_df.columns:
            aggregated_df["Abbrev"] = aggregated_df["Assigned_Outfall"].map(
                abbrev_map
            )
        results_df = add_scenario_cols(
            results_df, tag, task["Temperature"], task["SO4"], task["COD"]
        )
        aggregated_df = add_scenario_cols(
            aggregated_df, tag, task["Temperature"], task["SO4"], task["COD"]
        )
        return {
            "City": city_label,
            "TAG": tag,
            "results_df": results_df,
            "aggregated_df": aggregated_df,
            "all_abbrev_map": abbrev_map,
        }
    except Exception as exc:
        print(f"[Worker error] {city_label} / {tag}: {exc}", flush=True)
        gc.collect()
        return None


# ================================================================
# 6. PREPROCESSING / CACHE
# ================================================================
def preprocess_all_scenarios():
    os.makedirs(output_dir, exist_ok=True)
    tasks = [dict(sc, City=city) for sc in SCENARIOS for city in CITY_CONFIG]
    print(
        f"Figure 4 preprocessing: {len(tasks)} city-scenario tasks; "
        f"requested workers: {N_WORKERS}",
        flush=True,
    )
    t0 = time.time()

    if N_WORKERS <= 1:
        results = []
        for index, task in enumerate(tasks, start=1):
            results.append(_city_worker(task))
            print(
                f"[Figure4 {index:02d}/{len(tasks)}] "
                f"{task['City']} / {task['TAG']}",
                flush=True,
            )
    else:
        n_workers = max(1, min(int(N_WORKERS), cpu_count() - 1, len(tasks)))
        with Pool(n_workers) as pool:
            results = []
            for index, result in enumerate(
                pool.imap_unordered(_city_worker, tasks, chunksize=1), start=1
            ):
                results.append(result)
                label = "FAILED" if result is None else (
                    f"{result['City']} / {result['TAG']}"
                )
                print(
                    f"[Figure4 {index:02d}/{len(tasks)}] {label}",
                    flush=True,
                )

    print(f"\nProcessing finished in {time.time() - t0:.1f}s")
    valid = [r for r in results if r is not None]
    if len(valid) != len(tasks):
        raise RuntimeError(
            f"Figure 4 preprocessing incomplete: {len(valid)}/{len(tasks)} "
            "city-scenario tasks succeeded."
        )

    results_df = pd.concat([r["results_df"] for r in valid], ignore_index=True)
    aggregated_df = pd.concat([r["aggregated_df"] for r in valid], ignore_index=True)
    results_df["Q_UNIT"] = "m3/day"
    aggregated_df["Q_UNIT"] = "m3/day"

    abbrev_global = {}
    for r in valid:
        abbrev_global.update(r["all_abbrev_map"])

    if "total_length_km" not in aggregated_df.columns and "total_length" in aggregated_df.columns:
        aggregated_df["total_length_km"] = (
            pd.to_numeric(aggregated_df["total_length"], errors="coerce") / 1000.0
        )

    if SAVE_CACHE:
        aggregated_df.to_csv(CACHE_AGGREGATED_PATH, index=False, encoding="utf-8-sig")
        results_df.to_csv(CACHE_RESULTS_PATH, index=False, encoding="utf-8-sig")
        with open(CACHE_ABBREV_PATH, "w", encoding="utf-8") as f:
            json.dump(abbrev_global, f, ensure_ascii=False, indent=2)
        print(f"Cache saved:\n  {CACHE_AGGREGATED_PATH}\n  {CACHE_RESULTS_PATH}")

    gc.collect()
    return results_df, aggregated_df, abbrev_global


def load_cached_preprocessing():
    if not os.path.exists(CACHE_AGGREGATED_PATH):
        raise FileNotFoundError(CACHE_AGGREGATED_PATH)
    if not os.path.exists(CACHE_RESULTS_PATH):
        raise FileNotFoundError(CACHE_RESULTS_PATH)
    print("Loading cached preprocessing files...")
    aggregated_df = pd.read_csv(CACHE_AGGREGATED_PATH, low_memory=False)
    results_df = pd.read_csv(CACHE_RESULTS_PATH, low_memory=False)
    aggregated_df = ensure_q_units_m3_per_day(aggregated_df)
    results_df = ensure_q_units_m3_per_day(results_df)
    aggregated_df = add_building_sum_ratio_columns(aggregated_df)
    if "Catchment" in results_df.columns and "Assigned_Outfall" not in results_df.columns:
        results_for_ratio = results_df.rename(columns={"Catchment": "Assigned_Outfall"})
        results_for_ratio = add_building_sum_ratio_columns(results_for_ratio)
        results_df = results_for_ratio.rename(columns={"Assigned_Outfall": "Catchment"})
    if os.path.exists(CACHE_ABBREV_PATH):
        with open(CACHE_ABBREV_PATH, "r", encoding="utf-8") as f:
            abbrev_global = json.load(f)
    else:
        abbrev_global = {}
    return results_df, aggregated_df, abbrev_global


# ================================================================
# 7. PLOTTING DATAFRAMES
# ================================================================
def make_catchment_means_dataframe(aggregated_df):
    df = ensure_q_units_m3_per_day(aggregated_df.copy())
    numeric_cols = [
        "Problem_Length_km", "Toxic_Length_km", "Contaminated_Area_km2",
        "average_slope", "average_velocity", "average_HRT",
        "total_length_km", "average_AV", "flowrate_to_STW",
        "sum_L_build", "ave_L_build", "sum_L_build_over_Q", "sum_L_build_Q",
        "sum_Q_build", "sum_L_build_over_sum_Q_build",
        "average_sulphide", "average_methane", "average_do", "average_SO4",
        "total_flow",
    ]
    safe_numeric(df, numeric_cols)
    group_cols = ["City", "Assigned_Outfall"]
    if "Abbrev" in df.columns:
        group_cols.append("Abbrev")
    plot_df = df.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    plot_df["Q_UNIT"] = "m3/day"
    print(f"\nRF data: {len(plot_df)} catchment-level points")
    return plot_df


def make_catchment_means_with_uncertainty(aggregated_df):
    df = ensure_q_units_m3_per_day(aggregated_df.copy())
    numeric_cols = [
        "Problem_Length_km", "Toxic_Length_km", "Contaminated_Area_km2",
        "average_slope", "average_velocity", "average_HRT",
        "total_length_km", "average_AV", "flowrate_to_STW",
        "sum_L_build", "ave_L_build", "sum_L_build_over_Q", "sum_L_build_Q",
        "sum_Q_build", "sum_L_build_over_sum_Q_build",
        "average_sulphide", "average_methane", "average_do", "average_SO4",
        "total_flow",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    safe_numeric(df, numeric_cols)
    group_cols = ["City", "Assigned_Outfall"]
    if "Abbrev" in df.columns:
        group_cols.append("Abbrev")
    agg_dict = {c: ["mean", "std", "median", "min", "max", "count"]
                for c in numeric_cols}
    plot_df = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()
    new_cols = []
    for col in plot_df.columns:
        if isinstance(col, tuple):
            new_cols.append(col[0] if col[1] == "" else f"{col[0]}_{col[1]}")
        else:
            new_cols.append(col)
    plot_df.columns = new_cols
    plot_df["Q_UNIT"] = "m3/day"
    print(f"Catchment-mean (mean+SD) df: {len(plot_df)} catchments")
    return plot_df


# ================================================================
# 8. PLOTTING FUNCTIONS
# ================================================================
def feature_importance_panel(ax, df, feature_cols, risk, panel_tag):
    sub = df[feature_cols + [risk]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[sub[risk] >= 0].copy()
    valid_features = [c for c in feature_cols
                      if c in sub.columns and sub[c].nunique(dropna=True) > 1]
    if len(valid_features) < 1 or len(sub) < 8:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        ax.axis("off")
        return None

    X = sub[valid_features].values
    y = sub[risk].values
    rf = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1,
                               max_depth=None, min_samples_split=3)
    rf.fit(X, y)
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    sorted_names = [valid_features[i] for i in order]
    sorted_vals = imp[order]

    rhos, pvals = [], []
    for fn in sorted_names:
        s2 = sub[[fn, risk]].dropna()
        if len(s2) >= 4 and s2[fn].nunique() > 1:
            r, p = stats.spearmanr(s2[fn], s2[risk])
        else:
            r, p = np.nan, np.nan
        rhos.append(r); pvals.append(p)

    color = RISK_COLORS[risk]
    y_pos = np.arange(len(sorted_names))[::-1]
    bars = ax.barh(y_pos, sorted_vals, color=color, alpha=0.88,
                   edgecolor=SPINE_C, linewidth=0.35, height=0.58)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT_PARAM.get(n, n) for n in sorted_names], fontsize=6.2)
    ax.set_xlabel("Random-Forest importance", fontsize=7.4)
    ax.set_title(RISK_SHORT[risk], fontsize=8.6, color=color, fontweight="bold", pad=4)
    xmax = max(sorted_vals) * 1.45 if len(sorted_vals) else 1
    if xmax <= 0 or not np.isfinite(xmax):
        xmax = 1
    ax.set_xlim(0, xmax)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.45, color=GRID_C, alpha=0.8)
    ax.set_axisbelow(True)

    for b, rho, pv, val in zip(bars, rhos, pvals, sorted_vals):
        txt = f"{val:.2f}" if np.isnan(rho) else fr"{val:.2f}; $\rho$={rho:+.2f} {p_to_stars(pv)}"
        ax.text(val + xmax * 0.018, b.get_y() + b.get_height() / 2, txt,
                va="center", ha="left", fontsize=5.5, color=TEXT_C)

    panel_label(ax, panel_tag)
    return {
        "risk": risk,
        "features_ranked": sorted_names,
        "importances_ranked": sorted_vals.tolist(),
        "rhos_ranked": rhos,
        "pvals_ranked": pvals,
    }


def compute_loglog_ci_band(lx, ly, lx_line, alpha=0.05):
    n = len(lx)
    X = np.column_stack([np.ones(n), lx])
    beta = np.linalg.lstsq(X, ly, rcond=None)[0]
    pred = X @ beta
    resid = ly - pred
    dof = n - 2
    if dof <= 0:
        pred_line = beta[0] + beta[1] * lx_line
        return pred_line, pred_line, pred_line
    s2 = np.sum(resid ** 2) / dof
    x_mean = np.mean(lx)
    sxx = np.sum((lx - x_mean) ** 2)
    pred_line = beta[0] + beta[1] * lx_line
    if sxx <= 0:
        se_mean = np.full_like(lx_line, np.nan, dtype=float)
    else:
        se_mean = np.sqrt(s2 * (1.0 / n + ((lx_line - x_mean) ** 2) / sxx))
    tcrit = t_dist.ppf(1.0 - alpha / 2.0, dof)
    return pred_line, pred_line - tcrit * se_mean, pred_line + tcrit * se_mean


def risk_vs_top_parameter_panel_catchment_means(
        ax, df_unc, xcol, ycol, panel_tag,
        n_scenarios_kept=N_SCENARIOS_KEPT, audit_records=None):
    x_mean_col, x_std_col = f"{xcol}_mean", f"{xcol}_std"
    y_mean_col, y_std_col = f"{ycol}_mean", f"{ycol}_std"
    needed = [x_mean_col, x_std_col, y_mean_col, y_std_col, "City"]
    missing = [c for c in needed if c not in df_unc.columns]
    if missing:
        ax.text(0.5, 0.5, f"Missing cols:\n{missing}", transform=ax.transAxes,
                ha="center", va="center", fontsize=7)
        ax.axis("off"); panel_label(ax, panel_tag); return None

    identity_cols = [
        col for col in ("Assigned_Outfall", "Abbrev")
        if col in df_unc.columns
    ]
    raw = df_unc[needed + identity_cols].replace([np.inf, -np.inf], np.nan).copy()
    raw["plot_status"] = "included_positive"
    raw.loc[raw["City"].isna(), "plot_status"] = "excluded_missing_city"
    raw.loc[raw[x_mean_col].isna(), "plot_status"] = "excluded_missing_x"
    raw.loc[raw[y_mean_col].isna(), "plot_status"] = "excluded_missing_y"
    raw.loc[
        raw[x_mean_col].notna() & (raw[x_mean_col] <= 0), "plot_status"
    ] = "excluded_nonpositive_x_on_log_axis"
    raw.loc[
        raw[y_mean_col].notna() & (raw[y_mean_col] <= 0), "plot_status"
    ] = "excluded_nonpositive_y_on_log_axis"
    raw["exact_xy_group_size"] = 0
    raw["near_xy_group_size"] = 0
    included_mask = raw["plot_status"] == "included_positive"
    if included_mask.any():
        included_xy = raw.loc[included_mask, [x_mean_col, y_mean_col]].copy()
        exact_keys = pd.DataFrame({
            "x": included_xy[x_mean_col].round(12),
            "y": included_xy[y_mean_col].round(12),
        }, index=included_xy.index)
        near_keys = pd.DataFrame({
            "log10_x_rounded": np.log10(included_xy[x_mean_col]).round(2),
            "log10_y_rounded": np.log10(included_xy[y_mean_col]).round(2),
        }, index=included_xy.index)
        raw.loc[included_mask, "exact_xy_group_size"] = (
            exact_keys.groupby(["x", "y"])["x"].transform("size")
        )
        raw.loc[included_mask, "near_xy_group_size"] = (
            near_keys.groupby(["log10_x_rounded", "log10_y_rounded"])
            ["log10_x_rounded"].transform("size")
        )
    sub = raw[raw["plot_status"] == "included_positive"].copy()
    if audit_records is not None:
        for _, row in raw.iterrows():
            record = {
                "panel": panel_tag,
                "risk": ycol,
                "x_feature": xcol,
                "City": row.get("City"),
                "Assigned_Outfall": row.get("Assigned_Outfall"),
                "Abbrev": row.get("Abbrev"),
                "x_mean": row.get(x_mean_col),
                "y_mean": row.get(y_mean_col),
                "plot_status": row["plot_status"],
                "exact_xy_group_size": int(row["exact_xy_group_size"]),
                "near_xy_group_size": int(row["near_xy_group_size"]),
            }
            audit_records.append(record)
    excluded = len(raw) - len(sub)
    near_overlap_points = int((sub["near_xy_group_size"] > 1).sum())
    print(
        f"[Figure4 panel {panel_tag}] {ycol}: plotted {len(sub)}/{len(raw)} "
        f"catchments; excluded from log axes/fit: {excluded}; "
        f"points in near-overlap groups: {near_overlap_points}",
        flush=True,
    )
    if len(sub) < 4 or sub[x_mean_col].nunique() < 2:
        ax.text(0.5, 0.5, "Insufficient positive data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axis("off")
        panel_label(ax, panel_tag); return None

    color = RISK_COLORS[ycol]
    x = sub[x_mean_col].values.astype(float)
    y = sub[y_mean_col].values.astype(float)
    xerr = np.nan_to_num(sub[x_std_col].values.astype(float), nan=0.0)
    yerr = np.nan_to_num(sub[y_std_col].values.astype(float), nan=0.0)
    xerr_lower = np.minimum(xerr, x * 0.99)
    yerr_lower = np.minimum(yerr, y * 0.99)
    xerr_plot = np.array([xerr_lower, xerr])
    yerr_plot = np.array([yerr_lower, yerr])
    cities_arr = sub["City"].values

    for city in ["Hong Kong", "Toronto", "Los Angeles"]:
        m = cities_arr == city
        if not m.any():
            continue
        ax.errorbar(
            x[m], y[m], xerr=xerr_plot[:, m], yerr=yerr_plot[:, m],
            fmt=CITY_MARKERS[city], color=CITY_COLORS[city],
            ecolor=CITY_COLORS[city], elinewidth=0.45,
            capsize=1.2, capthick=0.45, markersize=5.0,
            markeredgecolor="white", markeredgewidth=0.4,
            alpha=0.9, label=city, zorder=5,
        )

    lx = np.log10(x); ly = np.log10(y)
    reg = LinearRegression().fit(lx.reshape(-1, 1), ly)
    b = reg.coef_[0]; a = 10 ** reg.intercept_
    pred_log = reg.predict(lx.reshape(-1, 1))

    x_min_data, x_max_data = x.min(), x.max()
    x_line = np.logspace(np.log10(x_min_data), np.log10(x_max_data), 200)
    lx_line = np.log10(x_line)
    pred_line_log, lower_log, upper_log = compute_loglog_ci_band(
        lx=lx, ly=ly, lx_line=lx_line, alpha=0.05,
    )
    y_line = 10 ** pred_line_log
    y_lower = 10 ** lower_log
    y_upper = 10 ** upper_log

    ax.fill_between(x_line, y_lower, y_upper, color=color,
                    alpha=0.16, linewidth=0, zorder=3)
    ax.plot(x_line, y_line, color=color, linewidth=1.3, alpha=0.95, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")

    ss_res = np.sum((ly - pred_log) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    n = len(sub); k = 1
    if n - k - 1 > 0 and ss_tot > 0 and ss_res >= 0:
        if ss_res == 0:
            p_model = 0.0
        else:
            f_stat = ((ss_tot - ss_res) / k) / (ss_res / (n - k - 1))
            p_model = 1 - f_dist.cdf(f_stat, k, n - k - 1)
    else:
        p_model = np.nan

    if len(sub) >= 4 and sub[x_mean_col].nunique() > 1:
        rho, pv_sp = stats.spearmanr(x, y)
    else:
        rho, pv_sp = np.nan, np.nan

    x_label = SHORT_PARAM.get(xcol, xcol)
    x_unit = UNIT_PARAM.get(xcol, "")
    xlabel = f"{x_label} ({x_unit})" if x_unit else x_label
    ax.set_xlabel(xlabel, fontsize=7.4)
    ax.set_ylabel(f"{RISK_LABEL[ycol]} ({RISK_UNIT[ycol]})", fontsize=7.4)
    ax.set_title(RISK_SHORT[ycol], fontsize=8.6, color=color, fontweight="bold", pad=4)
    ax.grid(True, linestyle=":", linewidth=0.45, color=GRID_C, alpha=0.8, which="both")
    ax.set_axisbelow(True)

    fit_line = format_power_law_equation(a, b)
    txt = (fit_line + "\n" + fr"$R^2={r2:.2f}$" + "\n"
           + format_p_annotation(p_model))
    ax.text(0.96, 0.04, txt, transform=ax.transAxes,
            fontsize=6.8, ha="right", va="bottom", color=TEXT_C,
            bbox=dict(boxstyle="round,pad=0.24", fc="white",
                      ec="#C8C8C8", lw=0.45, alpha=0.94),
            zorder=10)
    panel_label(ax, panel_tag)
    return {
        "risk": ycol, "x": xcol,
        "fit_type": "log_log_power_law_catchment_mean",
        "a": a, "b": b, "r2": r2, "n_catchments": n,
        "n_scenarios_per_catchment": n_scenarios_kept,
        "p": p_model, "spearman_rho": rho, "spearman_p": pv_sp,
        "stars": p_to_stars(pv_sp),
        "formula": f"{ycol} = {a:.6g} * {xcol}^{b:.4f}",
        "log_formula": f"log10({ycol}) = {reg.intercept_:.6g} + {b:.6g} * log10({xcol})",
        "x_train_min": float(x_min_data), "x_train_max": float(x_max_data),
    }


def flatten_importance_records(importance_records):
    rows = []
    for rec in importance_records:
        for rank, fn, im, rh, pv in zip(
            range(1, len(rec["features_ranked"]) + 1),
            rec["features_ranked"], rec["importances_ranked"],
            rec["rhos_ranked"], rec["pvals_ranked"],
        ):
            rows.append({
                "risk": rec["risk"], "rank": rank,
                "feature": fn, "feature_label": SHORT_PARAM.get(fn, fn),
                "importance": im, "spearman_rho": rh, "spearman_p": pv,
                "stars": p_to_stars(pv),
            })
    return pd.DataFrame(rows)


# ================================================================
# 9. DRAW FIGURE
# ================================================================
def draw_figure(results_df, aggregated_df, all_abbrev_map_global=None):
    os.makedirs(output_dir, exist_ok=True)
    results_df = ensure_q_units_m3_per_day(results_df)
    aggregated_df = ensure_q_units_m3_per_day(aggregated_df)

    results_df = filter_scenarios_keep(results_df)
    aggregated_df = filter_scenarios_keep(aggregated_df)

    plot_df = make_catchment_means_dataframe(aggregated_df)
    plot_df_unc = make_catchment_means_with_uncertainty(aggregated_df)
    if len(plot_df_unc) != EXPECTED_CATCHMENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CATCHMENTS} catchments before panel filtering, "
            f"but found {len(plot_df_unc)}. Check outfall assignment and inputs."
        )

    feature_pool = [
        "average_slope", "average_velocity", "average_HRT",
        "total_length_km", "average_AV", "flowrate_to_STW",
        "sum_L_build", "ave_L_build", "sum_L_build_over_Q", "sum_L_build_Q",
        "sum_Q_build", "sum_L_build_over_sum_Q_build",
        "average_sulphide", "average_methane", "average_do", "average_SO4",
        "total_flow",
    ]
    feature_pool = [c for c in feature_pool if c in plot_df.columns]
    feature_pool = [c for c in feature_pool
                    if plot_df[c].notna().sum() >= 5
                    and plot_df[c].nunique(dropna=True) > 1]

    fig = plt.figure(figsize=(7 * 1.3, 5.5 * 1.3), dpi=300)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           height_ratios=[2.3, 1.45],
                           hspace=0.35, wspace=0.5,
                           left=0.115, right=0.965, top=0.935, bottom=0.205)

    panel_letters = list("abcdef")
    pidx = 0
    importance_records = []
    for j, risk in enumerate(RISKS):
        ax = fig.add_subplot(gs[0, j])
        apply_axis_style(ax, half_frame=True)
        rec = feature_importance_panel(ax, plot_df, feature_pool,
                                       risk, panel_letters[pidx])
        pidx += 1
        if rec is not None:
            importance_records.append(rec)

    top_feature_by_risk = {
        rec["risk"]: rec["features_ranked"][0]
        for rec in importance_records
        if rec is not None and len(rec["features_ranked"]) > 0
    }

    formula_records_main = []
    point_audit_records = []
    for j, risk in enumerate(RISKS):
        ax = fig.add_subplot(gs[1, j])
        apply_axis_style(ax, half_frame=True)
        panel_tag = panel_letters[pidx]
        xcol = top_feature_by_risk.get(risk)
        if xcol is None:
            ax.text(0.5, 0.5, "Top parameter unavailable", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8)
            ax.axis("off"); panel_label(ax, panel_tag); pidx += 1; continue
        rec = risk_vs_top_parameter_panel_catchment_means(
            ax, plot_df_unc, xcol, risk, panel_tag, N_SCENARIOS_KEPT,
            audit_records=point_audit_records,
        )
        pidx += 1
        if rec is not None:
            formula_records_main.append(rec)

    city_handles = [
        Line2D([0], [0], marker=CITY_MARKERS[c], linestyle="None",
               markerfacecolor=CITY_COLORS[c], markeredgecolor="white",
               markeredgewidth=0.4, markersize=5.5,
               color=CITY_COLORS[c],
               label=f"{c} (mean ± SD across {N_SCENARIOS_KEPT} scenarios)")
        for c in ["Hong Kong", "Toronto", "Los Angeles"]
    ]
    fig.legend(handles=city_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.1),
               fontsize=6.2, frameon=False, ncol=3,
               handletextpad=0.5, labelspacing=0.45, borderaxespad=0.2)

    # ============================================================
    # 导出图件:高分辨率 PNG + 矢量图 (PDF / SVG)
    # ============================================================
    FIG_TAG = (
        f"v7_catchmentmeans_errbar_main_keep{N_SCENARIOS_KEPT}scen_"
        f"corr_gt_1mm_per_year_{BUILD_Q_MIN_TAG}"
    )
    # 确保矢量字体可编辑(嵌入字形而非转曲线/位图)
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    figure_prefix = "Figure4"
    base_path = os.path.join(output_dir, f"{figure_prefix}_{FIG_TAG}")

    # 1) 高分辨率 PNG (600 dpi)
    png_path = base_path + "_600dpi.png"
    fig.savefig(png_path, dpi=600, bbox_inches="tight",
                facecolor="white", transparent=False)

    # 2) 超高分辨率 PNG (1200 dpi,投稿/打印用,可选)

    # 3) PDF 矢量图(无限缩放不失真,期刊首选)

    # 4) SVG 矢量图(可在 Illustrator / Inkscape 中再编辑)

    print("\n=== 图件已导出 ===")
    print(f"  PNG  (600 dpi) : {png_path}")

    if NO_SHOW:
        plt.close(fig)
    else:
        plt.show()

    importance_long_df = flatten_importance_records(importance_records)
    importance_long_df.to_csv(
        os.path.join(output_dir, f"feature_importance_full_ranking_{FIG_TAG}.csv"),
        index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(formula_records_main).to_csv(
        os.path.join(output_dir, f"main_panels_def_loglog_formulas_{FIG_TAG}.csv"),
        index=False, encoding="utf-8-sig"
    )
    plot_df.to_csv(
        os.path.join(output_dir, f"plot_data_catchment_means_RF_{FIG_TAG}.csv"),
        index=False, encoding="utf-8-sig"
    )
    plot_df_unc.to_csv(
        os.path.join(output_dir, f"plot_data_catchment_means_with_uncertainty_{FIG_TAG}.csv"),
        index=False, encoding="utf-8-sig"
    )
    point_audit_df = pd.DataFrame(point_audit_records)
    point_audit_df.to_csv(
        os.path.join(output_dir, f"panel_def_point_audit_{FIG_TAG}.csv"),
        index=False, encoding="utf-8-sig"
    )
    if not point_audit_df.empty:
        point_counts = (
            point_audit_df.groupby(["panel", "risk", "plot_status"], dropna=False)
            .size().rename("n_catchments").reset_index()
        )
        point_counts.to_csv(
            os.path.join(output_dir, f"panel_def_point_count_summary_{FIG_TAG}.csv"),
            index=False, encoding="utf-8-sig"
        )
    aggregated_df.to_csv(
        os.path.join(output_dir, f"aggregated_kept_scenarios_{FIG_TAG}.csv"),
        index=False, encoding="utf-8-sig"
    )
    results_df.to_csv(
        os.path.join(output_dir, f"results_kept_scenarios_{FIG_TAG}.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 导出 Code2 / 后续代码用的回归方程
    reg_export_rows = []
    for r in formula_records_main:
        risk = r["risk"]; a = r["a"]; b = r["b"]
        reg_export_rows.append({
            "y_var": risk, "x_var": r["x"],
            "slope_log10": b, "intercept_log10": float(np.log10(a)),
            "a_linear": a, "b_exponent": b,
            "r2": r["r2"], "n_catchments": r["n_catchments"],
            "p": r["p"], "x_unit": "m day m-3",
            "y_unit": RISK_UNIT[risk], "formula": r["formula"],
            "x_train_min": r["x_train_min"], "x_train_max": r["x_train_max"],
            "scenario_grid": SCENARIO_GRID,
            "n_scenarios": N_SCENARIOS_KEPT,
            "building_q_min_m3_per_d_for_sum_L_over_Q": BUILD_Q_MIN_M3_PER_D,
            "corrosion_rate_threshold_mm_per_year": (
                CORROSION_RATE_THRESHOLD_MM_PER_YEAR
            ),
        })
    reg_export_df = pd.DataFrame(reg_export_rows)
    REG_TAG = "T_20_SO4_15_COD_525"
    reg_export_df.to_csv(
        os.path.join(output_dir, f"regressed_equations_{REG_TAG}.csv"),
        index=False, encoding="utf-8-sig"
    )

    print("\n=== Main figure d-f: log-log regression ===")
    for r in formula_records_main:
        print(f"  {r['risk']:<28s} | x={r['x']:<22s} "
              f"n={r['n_catchments']}, R²={r['r2']:.3f}, "
              f"p={format_p_value(r['p'])}, b={r['b']:.3f}")
        print(f"      formula: {r['formula']}")
    print(f"\nOK All outputs saved to: {output_dir}")


# ================================================================
# 10. MAIN
# ================================================================
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    if RUN_PREPROCESSING:
        print("\nMode: preprocessing + figure drawing")
        results_df, aggregated_df, abbrev_global = preprocess_all_scenarios()
    else:
        print("\nMode: figure drawing only (loading cache)")
        results_df, aggregated_df, abbrev_global = load_cached_preprocessing()

    draw_figure(results_df, aggregated_df, abbrev_global)
