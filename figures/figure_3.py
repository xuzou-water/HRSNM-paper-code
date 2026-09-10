# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:24:22 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:37:00 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:19:36 2026

@author: zouxu

Figure 3 (final; single figure, 4 columns x 3 rows; panels a-l).
All diameter-resolved panels show gravity sewers only.
"""

import argparse
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = SCRIPT_DIR / "node_dwf_fix_results"
OUT_DIR = str(RESULTS_ROOT / "figure3")
DPI = 600


def parse_args():
    parser = argparse.ArgumentParser(description="Generate final HRSNM Figure 3.")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "figure3")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()

SHOW_N_LABELS = False
X_MIN, X_MAX = 0, 2500       # 横坐标范围 (mm)

# ── 开关：figure a-d 纵坐标是否使用对数坐标 ──────────────────────────
LOG_AD = False                # True -> a-d 使用 log 坐标; False -> 线性坐标

# ── 开关：a-d / i-l 散点是否区分重力管 / 压力管 ──────────────────
SPLIT_PIPE_TYPE = True

# ── 开关：打开后先剔除所有压力管(rising_main)，再计算/绘图 ───────────
EXCLUDE_RISING_MAIN = True   # True -> 全程只用重力管，压力管被丢弃

CORR_METHOD = "partial_spearman"   # or "spearman"
MIN_OBS = 50

# Upstream / midstream / downstream diameter boundaries (mm).
UP_MAX = 500
MID_MAX = 1200
DOWN_MIN = 1200

# ════════════════════════════════════════════════════════════════════
# STYLE
# ════════════════════════════════════════════════════════════════════
try:
    plt.rcParams["font.family"] = "Arial"
except Exception:
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["axes.unicode_minus"] = False

BG_FIG = "white"
BG_AX = "white"
GRID_C = "#D9D4CB"
SPINE_C = "#333333"
TEXT_C = "#222222"
SUBTEXT_C = "#666666"

SAGE = "#A1B4AC"
TEAL = "#3A7A8C"
NAVY = "#08345A"
BEIGE = "#E7E3DA"
BROWN = "#A66842"

# Very light neutral fills distinguish network position without competing with
# the city colours. Boundary lines provide a non-colour cue at 500/1,200 mm.
ZONE_SPECS = (
    (0, UP_MAX, "Upstream (<500 mm)", "#F1F5F3"),
    (UP_MAX, MID_MAX, "Midstream (500–<1,200 mm)", "#F8F4EF"),
    (DOWN_MIN, X_MAX, "Downstream (1,200–2,500 mm)", "#F2F3F6"),
)
ZONE_BOUNDARY_C = "#AAA49B"

plt.rcParams.update({
    "font.size": 7,
    "axes.linewidth": 0.7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.8,
    "legend.fontsize": 6.2,
    "legend.frameon": False,
    "figure.facecolor": BG_FIG,
    "axes.facecolor": BG_AX,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "axes.edgecolor": SPINE_C,
    "axes.labelcolor": TEXT_C,
    "xtick.color": TEXT_C,
    "ytick.color": TEXT_C,
    "text.color": TEXT_C,
})

CITY_COLORS = {"Hong Kong": TEAL, "Toronto": BROWN, "Los Angeles": SAGE}
CITY_MARKERS = {"Hong Kong": "D", "Toronto": "o", "Los Angeles": "s"}

# ── 管型样式：gravity = 重力管, rising_main = 压力管 ────────────────
PIPE_TYPES = ["gravity", "rising_main"]
PIPE_TYPE_LABELS = {"gravity": "Gravity (重力管)",
                    "rising_main": "Rising main (压力管)"}
PIPE_TYPE_STYLE = {
    "gravity":     dict(filled=True,  linestyle="--"),
    "rising_main": dict(filled=False, linestyle=":"),
}

POS_COLOR = BROWN
NEG_COLOR = NAVY

RHO_LABEL = (r"Partial Spearman $\rho$"
             if CORR_METHOD == "partial_spearman"
             else r"Spearman $\rho$")

# ── 标准规格识别 / 分组参数 ────────────────────────────────────────
MIN_N = 30
STD_REL_TOL = 0.05
STD_ROUND = 0

# All cities use exactly the same user-selected target diameter classes.
SELECTED_DIAMETERS = np.array([
    225, 300, 375, 450, 525, 600, 675, 750, 900,
    1200, 1500, 1800, 2100, 2400
], dtype=float)
FIXED_STANDARDS = {
    city: SELECTED_DIAMETERS.copy()
    for city in ("Hong Kong", "Toronto", "Los Angeles")
}
COMMON_PLOT_DIAMETERS = SELECTED_DIAMETERS.copy()

# Exclude pipes below the lower edge of the 225-mm target class, preventing
# Hong Kong's 100/150-mm pipes from being folded into that class. Toronto/LA
# 200/250-mm pipes are represented by the shared 225-mm target class.
BIN_RAW_MIN = 187.5
BIN_RAW_MAX = 2550.0

# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_pipe_type(series):
    """把 pipe_type 列标准化为 'gravity' / 'rising_main' / 其他原值(小写)。"""
    s = series.astype(str).str.strip().str.lower()
    s = s.replace({
        "gravity sewer": "gravity",
        "gravity_sewer": "gravity",
        "rising main": "rising_main",
        "risingmain": "rising_main",
        "pressure": "rising_main",
        "pressure_main": "rising_main",
        "force_main": "rising_main",
        "force main": "rising_main",
    })
    return s
def drop_rising_mains(data):
    """删除所有 pipe_type == 'rising_main' 的记录，返回过滤后的 data。"""
    new_data = {}
    for city, df in data.items():
        n0 = len(df)
        if "pipe_type" in df.columns:
            d = df[df["pipe_type"] != "rising_main"].copy()
        else:
            d = df.copy()
        n1 = len(d)
        print(f"  [{city}] rising mains removed: {n0 - n1:,} "
              f"-> remaining (gravity/unknown) = {n1:,}")
        if n1 > 0:
            new_data[city] = d
        else:
            print(f"  [{city}] no gravity data left, city dropped.")
    if len(new_data) == 0:
        raise ValueError("剔除压力管后没有任何可用数据，请检查 pipe_type 列。")
    return new_data

def apply_axis_style(ax, facecolor=BG_AX, half_frame=True):
    ax.set_facecolor(facecolor)
    if half_frame:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
    else:
        for s in ax.spines.values():
            s.set_visible(True)
    for s in ax.spines.values():
        s.set_color(SPINE_C)
        s.set_linewidth(0.7)
    ax.tick_params(axis="both", colors=TEXT_C, labelsize=6.5,
                   direction="in", width=0.7)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color(TEXT_C)


# ════════════════════════════════════════════════════════════════════
# 标准管径规格识别
# ════════════════════════════════════════════════════════════════════
def find_standard_diameters(diam, rel_tol=STD_REL_TOL,
                            min_count=MIN_N, round_decimals=STD_ROUND):
    s = pd.Series(diam).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.array([])
    counts = s.round(round_decimals).value_counts().sort_index()
    vals = counts.index.values.astype(float)
    cnts = counts.values.astype(float)
    clusters = []
    for v, c in zip(vals, cnts):
        if clusters and (v - clusters[-1]["max"]) <= rel_tol * clusters[-1]["peak_val"]:
            cl = clusters[-1]
            cl["count"] += c
            cl["max"] = v
            if c > cl["peak_count"]:
                cl["peak_count"] = c
                cl["peak_val"] = v
        else:
            clusters.append(dict(count=c, max=v, peak_val=v, peak_count=c))
    standards = [cl["peak_val"] for cl in clusters if cl["count"] >= min_count]
    return np.array(sorted(standards), dtype=float)


def compute_city_standards(data, x_col="diameter"):
    std = {}
    for city, df in data.items():
        if city in FIXED_STANDARDS:
            s = np.array(sorted(FIXED_STANDARDS[city]), dtype=float)
            s = s[(s >= X_MIN) & (s <= X_MAX)]
            std[city] = s
            if len(s):
                print(f"  [{city}] FIXED standard diameters (mm): "
                      f"{', '.join(f'{v:.0f}' for v in s)}")
            continue
        if x_col not in df.columns:
            std[city] = np.array([])
            continue
        s = find_standard_diameters(df[x_col].values)
        s = s[(s >= X_MIN) & (s <= X_MAX)]
        std[city] = s
        if len(s):
            print(f"  [{city}] standard diameters (mm): "
                  f"{', '.join(f'{v:.0f}' for v in s)}")
        else:
            print(f"  [{city}] no standard diameters detected.")
    return std


# ════════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════════
def load_data():
    paths = {
        "Los Angeles": dict(
            rp=str(RESULTS_ROOT / "LA_v3" / "biochemical_results" / "la_result_segments_v7.csv"),
            conc=str(RESULTS_ROOT / "LA_v3" / "biochemical_results" / "la_emission_radius_segments_v7.csv"),
        ),
        "Toronto": dict(
            rp=str(RESULTS_ROOT / "toronto_v3" / "biochemical_results" / "toronto_result_segments_v7.csv"),
            conc=str(RESULTS_ROOT / "toronto_v3" / "biochemical_results" / "toronto_emission_radius_segments_v7.csv"),
        ),
        "Hong Kong": dict(
            rp=str(RESULTS_ROOT / "HK_v3" / "biochemical_results" / "hk_result_segments_v7.csv"),
            conc=str(RESULTS_ROOT / "HK_v3" / "biochemical_results" / "hk_emission_radius_segments_v7.csv"),
        ),
    }

    rp_cols = {
        "name", "WIDTH", "v", "A_V", "A_V_g", "HRT",
        "diameter", "depth", "slope", "flowrate", "Vg", "Vw",
        "SHS_in", "SO_in", "SSO4_in", "SF_in", "Sac_in", "XHw_in",
        "SHS_out", "SO_out", "SF_out", "SSO4_out", "Sac_out", "ra", "Xs1_in",
        "R5", "R6", "R7", "rs2c", "rs2b", "rs2_ox_f", "pipe_type"
    }
    conc_cols = {"pipe_name", "SH2S", "dcorr_dt", "distance",
                 "gas_velocity", "ra", "pipe_type"}

    data = {}
    for city, p in paths.items():
        if (not os.path.exists(p["rp"])) or (not os.path.exists(p["conc"])):
            print(f"[Warning] Missing file(s) for {city}, skipped.")
            continue

        rp0 = pd.read_csv(p["rp"])
        conc0 = pd.read_csv(p["conc"])

        rp = rp0[[c for c in rp0.columns if c in rp_cols]].copy()
        conc = conc0[[c for c in conc0.columns if c in conc_cols]].copy()

        safe_numeric(rp, [
            "WIDTH", "v", "A_V", "A_V_g", "HRT", "diameter", "depth",
            "slope", "flowrate", "Vg", "Vw", "SHS_in", "SO_in",
            "SSO4_in", "SF_in", "Sac_in", "XHw_in",
            "SHS_out", "SO_out", "SF_out", "SSO4_out", "Sac_out", "ra",
            "Xs1_in", "R5", "R6", "R7", "rs2c", "rs2b", "rs2_ox_f",
        ])
        safe_numeric(conc, ["SH2S", "dcorr_dt", "distance",
                            "gas_velocity", "ra"])

        if "name" not in rp.columns or "pipe_name" not in conc.columns:
            print(f"[Warning] Missing merge key for {city}, skipped.")
            continue

        df = pd.merge(rp, conc, left_on="name", right_on="pipe_name",
                      how="inner", suffixes=("", "_conc"))

        if "diameter" not in df.columns:
            print(f"[Warning] diameter not found for {city}, skipped.")
            continue

        # ── 标准化 pipe_type 列 ──
        st_col = None
        if "pipe_type" in df.columns:
            st_col = "pipe_type"
        elif "pipe_type_conc" in df.columns:
            st_col = "pipe_type_conc"
        if st_col is not None:
            df["pipe_type"] = normalize_pipe_type(df[st_col])
            vc = df["pipe_type"].value_counts()
            print(f"  [{city}] pipe_type counts: "
                  + ", ".join(f"{k}={v}" for k, v in vc.items()))
        else:
            df["pipe_type"] = np.nan
            print(f"  [{city}] WARNING: no pipe_type column found.")

        df = df.dropna(subset=["diameter"]).copy()
        df["diameter"] = df["diameter"] * 1000.0   # m → mm

        # Relative water depth: h/D, where h is water depth and D is diameter.
        if {"depth", "diameter"}.issubset(df.columns):
            water_depth_mm = df["depth"] * 1000.0
            diameter_mm = df["diameter"]
            valid = (
                water_depth_mm.notna()
                & diameter_mm.notna()
                & (diameter_mm > 0)
                & (water_depth_mm >= 0)
                & (water_depth_mm <= diameter_mm)
            )
            df["h_D"] = np.where(valid, water_depth_mm / diameter_mm, np.nan)
        else:
            df["h_D"] = np.nan

        # Gas-volume-to-exposed-area ratio used directly by the corrosion model.
        if "A_V_g" in df.columns:
            df["Vg_Ag"] = (
                1.0 / df["A_V_g"].replace(0, np.nan)
            ).replace([np.inf, -np.inf], np.nan)
        else:
            df["Vg_Ag"] = np.nan

        # net reaction rate (保留计算，不再绘制)
        react_pos = ["R5", "R6", "R7"]
        react_neg = ["rs2c", "rs2b", "rs2_ox_f"]
        if set(react_pos + react_neg).issubset(df.columns):
            df["net_rate"] = (df[react_pos].sum(axis=1, skipna=True)
                              - df[react_neg].sum(axis=1, skipna=True))
        else:
            df["net_rate"] = np.nan

        data[city] = df
        print(f"  [{city}] n = {len(df):,}")

    if len(data) == 0:
        raise FileNotFoundError(
            "No valid input files were loaded. Please check BASE and filenames.")
    return data


# ════════════════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════════════════
def _binned_median_iqr_standard(df, x_col, y_col, standards, min_n=MIN_N):
    if (x_col not in df.columns or y_col not in df.columns
            or standards is None or len(standards) == 0):
        return (np.array([]),) * 5
    v = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    v = v[(v[x_col] >= BIN_RAW_MIN) & (v[x_col] < BIN_RAW_MAX)]
    if len(v) == 0:
        return (np.array([]),) * 5
    x = v[x_col].values.astype(float)
    y = v[y_col].values.astype(float)
    standards = np.asarray(sorted(standards), dtype=float)
    diffs = np.abs(x[:, None] - standards[None, :])
    nearest = np.argmin(diffs, axis=1)
    cx, median, p25, p75, nc = [], [], [], [], []
    for b in range(len(standards)):
        m = nearest == b
        n = int(m.sum())
        if n < min_n:
            continue
        yb = y[m]
        q25, q50, q75 = np.percentile(yb, [25, 50, 75])
        cx.append(standards[b]); median.append(q50)
        p25.append(q25); p75.append(q75)
        nc.append(n)
    return [np.array(a) for a in (cx, median, p25, p75, nc)]


def partial_spearman(df, x, y, covars, min_obs=MIN_OBS):
    covars = [c for c in covars if c in df.columns and c not in (x, y)]
    cols = [x, y] + covars
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    covars = [c for c in covars if sub[c].nunique() > 1]
    n = len(sub)
    if n < max(min_obs, len(covars) + 10):
        return np.nan, n
    if sub[x].nunique() <= 1 or sub[y].nunique() <= 1:
        return np.nan, n
    rx = rankdata(sub[x].values)
    ry = rankdata(sub[y].values)
    if covars:
        Z = np.column_stack([np.ones(n)] + [rankdata(sub[c].values) for c in covars])
        bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
        by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
        ex = rx - Z @ bx
        ey = ry - Z @ by
    else:
        ex = rx - rx.mean()
        ey = ry - ry.mean()
    if ex.std() == 0 or ey.std() == 0:
        return np.nan, n
    return float(np.corrcoef(ex, ey)[0, 1]), n


def compute_importance(data, outcomes, features, method=CORR_METHOD):
    out = {o: {} for o in outcomes}
    per_city = {o: {f: {} for f in features} for o in outcomes}
    for o in outcomes:
        for f in features:
            if f == o:
                continue
            rhos = []
            for city, df in data.items():
                if f not in df.columns or o not in df.columns:
                    continue
                if method == "partial_spearman":
                    covars = [c for c in features if c not in (f, o)]
                    rho, _ = partial_spearman(
                        df.replace([np.inf, -np.inf], np.nan), f, o, covars)
                    if not np.isfinite(rho):
                        continue
                else:
                    d = df[[f, o]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(d) < MIN_OBS or d[f].nunique() <= 1 or d[o].nunique() <= 1:
                        continue
                    rho, _ = spearmanr(d[f], d[o])
                    if not np.isfinite(rho):
                        continue
                rhos.append(rho)
                per_city[o][f][city] = rho
            if rhos:
                out[o][f] = np.mean(rhos)
    return out, per_city


# ════════════════════════════════════════════════════════════════════
# PLOT PRIMITIVES
# ════════════════════════════════════════════════════════════════════
def _fmt(ax):
    apply_axis_style(ax, facecolor=BG_AX, half_frame=True)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, color=GRID_C, alpha=0.45)
    ax.set_axisbelow(True)


def _label(ax, txt):
    x_pos = -0.18 if txt == "g" else -0.10
    ax.text(x_pos, 1.03, txt, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", ha="left", va="bottom", color=TEXT_C)


def plot_width_panel(ax, data, city_standards, y_col, ylabel,
                     show_n=False, log_y=False, split_type=SPLIT_PIPE_TYPE):
    # Encode network position consistently in every diameter-resolved panel.
    for xmin, xmax, _, color in ZONE_SPECS:
        ax.axvspan(xmin, xmax, facecolor=color, edgecolor="none",
                   alpha=0.78, zorder=0)
    for boundary in (UP_MAX, MID_MAX):
        ax.axvline(boundary, color=ZONE_BOUNDARY_C, linewidth=0.55,
                   linestyle=(0, (2.0, 2.0)), zorder=0.5)

    has_any = False
    # 记录所有正的数据点，便于设置 log 坐标下限
    pos_vals = []
    for ci, city in enumerate(["Hong Kong", "Toronto", "Los Angeles"]):
        if city not in data:
            continue
        df = data[city]
        if y_col not in df.columns:
            continue
        clr = CITY_COLORS[city]
        mkr = CITY_MARKERS[city]
        standards = city_standards.get(city, np.array([]))

        # ── 根据 pipe_type 拆分子集 ──
        if (split_type and "pipe_type" in df.columns
                and df["pipe_type"].notna().any()):
            subsets = []
            for stype in PIPE_TYPES:
                dsub = df[df["pipe_type"] == stype]
                if len(dsub) > 0:
                    subsets.append((stype, dsub))
            # 若没有任何已知类型，则退回到整体绘制
            if not subsets:
                subsets = [("gravity", df)]
        else:
            subsets = [("gravity", df)]

        for stype, dsub in subsets:
            sty = PIPE_TYPE_STYLE.get(stype, PIPE_TYPE_STYLE["gravity"])
            xc, ym, lo, hi, nc = _binned_median_iqr_standard(
                dsub, "diameter", y_col, standards)
            if len(xc) == 0:
                continue
            keep = ((xc >= X_MIN) & (xc <= X_MAX)
                    & np.isin(xc, COMMON_PLOT_DIAMETERS))
            xc, ym, lo, hi, nc = xc[keep], ym[keep], lo[keep], hi[keep], nc[keep]
            if len(xc) == 0:
                continue
            has_any = True

            # log 坐标下处理置信区间下界为非正值的情况
            lo_plot = lo.copy()
            if log_y:
                pos_vals.extend(ym[ym > 0].tolist())
                pos_vals.extend(lo[lo > 0].tolist())
                pos_vals.extend(hi[hi > 0].tolist())
                tiny = np.where(ym > 0, ym * 1e-3, 1e-6)
                lo_plot = np.where(lo_plot <= 0, tiny, lo_plot)

            ax.fill_between(xc, lo_plot, hi, color=clr, alpha=0.12,
                            linewidth=0, zorder=1)
            # 折线：重力管虚线，压力管点线，线宽 0.65
            ax.plot(xc, ym, color=clr, linewidth=0.65,
                    linestyle=sty["linestyle"], zorder=3)
            ms = 10 + 18 * np.log10(np.maximum(nc, 1) / max(nc.min(), 1))
            if sty["filled"]:
                # 重力管：实心标记
                ax.scatter(xc, ym, s=ms*0.5, color=clr, marker=mkr,
                           edgecolors="white", linewidths=0.45,
                           zorder=4, alpha=0.95)
            else:
                # 压力管：空心标记
                ax.scatter(xc, ym, s=ms*0.5, facecolors="white", edgecolors=clr,
                           marker=mkr, linewidths=0.8,
                           zorder=5, alpha=0.95)
            if show_n:
                y_offset = 5 + ci * 6
                for xi, yi, ni in zip(xc, ym, nc):
                    ax.annotate(f"n={int(ni)}", xy=(xi, yi),
                                textcoords="offset points",
                                xytext=(0, y_offset), ha="center", va="bottom",
                                fontsize=4.3, color=clr, zorder=6, clip_on=True)
    if not has_any:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color=SUBTEXT_C)
    ax.set_xlabel("Pipe diameter (mm)", fontsize=7.5, color=TEXT_C)
    ax.set_ylabel(ylabel, fontsize=7.5, color=TEXT_C)
    ax.set_xlim(X_MIN, X_MAX)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(250))
    _fmt(ax)

    # ── 设置对数纵坐标 ──
    if log_y and has_any and len(pos_vals) > 0:
        ax.set_yscale("log")
        ymin = max(min(pos_vals) * 0.6, 1e-9)
        ymax = max(pos_vals) * 1.6
        if ymax > ymin:
            ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def plot_importance_panel(ax, importance, per_city, outcome, feature_labels, title_txt):
    feats = [f for f in feature_labels.keys()
             if (f in importance.get(outcome, {}))
             or (len(per_city.get(outcome, {}).get(f, {})) > 0)]
    vals = np.array([importance[outcome].get(f, 0.0) for f in feats], dtype=float)
    labs = np.array([feature_labels[f] for f in feats])
    if len(vals) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color=SUBTEXT_C)
        ax.set_title(title_txt, fontsize=7.8, pad=4, color=TEXT_C)
        _fmt(ax)
        return
    order = np.argsort(np.abs(vals))
    vals = vals[order]; labs = labs[order]; feats = [feats[i] for i in order]
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, alpha=0.88, edgecolor="white",
            linewidth=0.35, height=0.72, zorder=2)
    for yi, feat in zip(y, feats):
        cd = per_city[outcome].get(feat, {})
        for city in ["Hong Kong", "Toronto", "Los Angeles"]:
            if city not in cd:
                continue
            ax.scatter(cd[city], yi, s=16, color=CITY_COLORS[city],
                       marker=CITY_MARKERS[city], edgecolors="white",
                       linewidths=0.35, zorder=4, alpha=0.98)
    ax.axvline(0, color=SPINE_C, linewidth=0.6, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=6.2, color=TEXT_C)
    ax.set_xlim(-1, 1)
    ax.set_xlabel(RHO_LABEL, fontsize=7.5, color=TEXT_C)
    ax.set_title(title_txt, fontsize=7.8, pad=4, color=TEXT_C)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    _fmt(ax)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, color=GRID_C, alpha=0.55)
    top = np.argmax(np.abs(vals))
    ax.text(0.98, 0.04, f"top: {labs[top]}\nrho = {vals[top]:+.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color=TEXT_C,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=BEIGE,
                      edgecolor="#B8B1A6", linewidth=0.45, alpha=0.95))





# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    args = parse_args()
    RESULTS_ROOT = args.results_root.resolve()
    OUT_DIR = str(args.output_dir.resolve())

    print("=" * 60)
    print(f"Correlation method: {CORR_METHOD}")
    print(f"Log y-axis for panels a-d: {LOG_AD}")
    print(f"Split gravity / rising_main: {SPLIT_PIPE_TYPE}")
    print("Loading baseline data ...")
    data = load_data()

    if EXCLUDE_RISING_MAIN:
        data = drop_rising_mains(data)
        SPLIT_PIPE_TYPE = False  

    print("Detecting standard pipe diameters per city ...")
    city_standards = compute_city_standards(data, x_col="diameter")
    print("  Selected shared diameters (mm): "
          + ", ".join(f"{v:.0f}" for v in COMMON_PLOT_DIAMETERS))


    FEATS = {
        "SHS_in":   r"$S_{HS,in}$",
        "SO_in":    r"$S_{O,in}$",
        "SSO4_in":  r"$S_{SO_4,in}$",
        "flowrate": r"Flow rate",
        "slope":    r"Slope",
        "A_V":      r"$A/V$",
        "Vg_Ag":    r"$V_g/A_g$",
        "v":        r"Velocity $v$",
        "h_D":      r"$h/D$"
    }
    OUTCOMES = ["SHS_in", "SH2S", "distance", "dcorr_dt"]

    print(f"Computing {CORR_METHOD} importance ...")
    importance, per_city = compute_importance(
        data, OUTCOMES, list(FEATS.keys()), method=CORR_METHOD)

    print("Building figure (4 columns x 3 rows; panels a-l) ...")
    fig = plt.figure(figsize=(12.0/1.3, 8.9/1.5), dpi=300)
    fig.patch.set_facecolor(BG_FIG)
    gs = fig.add_gridspec(3, 4, hspace=0.50, wspace=0.34,
                          left=0.07, right=0.985, top=0.965, bottom=0.175)

    # ── Row 1: a-d ──
    ax_a = fig.add_subplot(gs[0, 0]); ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2]); ax_d = fig.add_subplot(gs[0, 3])
    plot_width_panel(ax_a, data, city_standards, "SHS_in",
                     r"$S_{HS,in}$ (g m$^{-3}$)", SHOW_N_LABELS, log_y=LOG_AD)
    ax_a.set_ylim(0.0, 2.0)
    ax_a.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    plot_width_panel(ax_b, data, city_standards, "SH2S",
                     r"H$_2$S gas (ppm)", SHOW_N_LABELS, log_y=LOG_AD)
    plot_width_panel(ax_c, data, city_standards, "distance",
                     "Transport\ndistance (m)", SHOW_N_LABELS, log_y=LOG_AD)
    plot_width_panel(ax_d, data, city_standards, "dcorr_dt",
                     "Corrosion rate\n(mm yr$^{-1}$)", SHOW_N_LABELS, log_y=LOG_AD)
    _label(ax_a, "a"); _label(ax_b, "b"); _label(ax_c, "c"); _label(ax_d, "d")

    # ── Row 2: e-h (drivers) ──
    ax_e = fig.add_subplot(gs[1, 0]); ax_f = fig.add_subplot(gs[1, 1])
    ax_g = fig.add_subplot(gs[1, 2]); ax_h = fig.add_subplot(gs[1, 3])
    plot_importance_panel(ax_e, importance, per_city, "SHS_in", FEATS,
                          r"Driver of $S_{HS}$")
    plot_importance_panel(ax_f, importance, per_city, "SH2S", FEATS,
                          r"Driver of H$_2$S gas")
    plot_importance_panel(ax_g, importance, per_city, "distance", FEATS,
                          r"Driver of dispersion impact radius")
    plot_importance_panel(ax_h, importance, per_city, "dcorr_dt", FEATS,
                          r"Driver of corrosion rate")
    _label(ax_e, "e"); _label(ax_f, "f"); _label(ax_g, "g"); _label(ax_h, "h")

    # ── Row 3: i-l ──
    ax_i = fig.add_subplot(gs[2, 0]); ax_j = fig.add_subplot(gs[2, 1])
    ax_k = fig.add_subplot(gs[2, 2]); ax_l = fig.add_subplot(gs[2, 3])
    plot_width_panel(ax_i, data, city_standards, "SO_in",
                     r"$S_{O,in}$ (g m$^{-3}$)", SHOW_N_LABELS)
    plot_width_panel(ax_j, data, city_standards, "slope",
                     r"Slope", SHOW_N_LABELS)
    plot_width_panel(ax_k, data, city_standards, "A_V",
                     r"$A/V$ ratio", SHOW_N_LABELS)
    plot_width_panel(ax_l, data, city_standards, "h_D",
                     r"$h/D$", SHOW_N_LABELS, log_y=False)
    _label(ax_i, "i"); _label(ax_j, "j"); _label(ax_k, "k"); _label(ax_l, "l")

    # ── shared legend ──
    city_handles = [
        Line2D([], [], color=CITY_COLORS[c], marker=CITY_MARKERS[c],
               markersize=5.4, markeredgecolor="white", markeredgewidth=0.4,
               linewidth=0.65, linestyle="--", label=c)
        for c in ["Hong Kong", "Toronto", "Los Angeles"] if c in data
    ]
    # 管型图例：实心=重力管(虚线)，空心=压力管(点线)
    pipe_handles = [
        Line2D([], [], color=TEXT_C, marker="o", linestyle="--",
               markersize=5.4, markerfacecolor=TEXT_C, markeredgecolor="white",
               markeredgewidth=0.4, linewidth=0.65,
               label=PIPE_TYPE_LABELS["gravity"]),
        Line2D([], [], color=TEXT_C, marker="o", linestyle=":",
               markersize=5.4, markerfacecolor="white", markeredgecolor=TEXT_C,
               markeredgewidth=0.8, linewidth=0.65,
               label=PIPE_TYPE_LABELS["rising_main"]),
    ]
    sign_handles = [
        Line2D([], [], color=POS_COLOR, marker="s", linestyle="",
               markersize=5.8, label=r"positive $\rho$"),
        Line2D([], [], color=NEG_COLOR, marker="s", linestyle="",
               markersize=5.8, label=r"negative $\rho$"),
    ]
    handles = city_handles[:]
    if SPLIT_PIPE_TYPE:
        handles += pipe_handles
    handles += sign_handles
    leg = fig.legend(handles=handles,
                     loc="lower center",
                     bbox_to_anchor=(0.5, 0.052), ncol=len(handles),
                     frameon=False, fontsize=6.5, handlelength=1.4,
                     columnspacing=2.2, handletextpad=0.7)
    for t in leg.get_texts():
        t.set_color(TEXT_C)

    zone_handles = [
        Patch(facecolor=color, edgecolor=ZONE_BOUNDARY_C, linewidth=0.45,
              label=label)
        for _, _, label, color in ZONE_SPECS
    ]
    zone_leg = fig.legend(
        handles=zone_handles, loc="lower center", bbox_to_anchor=(0.5, 0.012),
        ncol=3, frameon=False, fontsize=6.3, handlelength=1.5,
        columnspacing=2.4, handletextpad=0.7,
    )
    for t in zone_leg.get_texts():
        t.set_color(TEXT_C)

    ensure_dir(OUT_DIR)
    out_png = os.path.join(OUT_DIR, "Figure3.png")
    fig.savefig(out_png, dpi=DPI, facecolor=BG_FIG, transparent=False)

    print("\n✓ Figure saved:")
    print(f"    {out_png} (600 dpi PNG)")




    if not args.no_show:
        plt.show()
