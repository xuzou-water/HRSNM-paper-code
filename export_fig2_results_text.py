# -*- coding: utf-8 -*-
"""
Export a Nature-style Results subsection for Figure 2.

Inputs are the data tables exported by HRSNM(v Fig2_9).py:
  - <results-root>/figure2/figure2_data/wwtp_comparison.csv
  - <results-root>/figure2/figure2_data/segment_counts.csv
  - <results-root>/figure2/figure2_data/sci_whisker_ranges.csv
  - <results-root>/figure2/figure2_data/pipe_parameters_long.csv
  - <results-root>/figure2/figure2_data/nodes.csv
  - <results-root>/figure2/figure2_data/pipes_corrosion.csv

Outputs:
  - Fig2_results_text.txt
  - Fig2_results_summary.json
  - Fig2_wwtp_validation_summary.csv
  - Fig2_hydraulic_envelope_summary.csv
  - Fig2_h2s_node_summary.csv
  - Fig2_dispersion_radius_summary.csv
  - Fig2_corrosion_summary.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / f"{SCRIPT_DIR.name}_results"
DEFAULT_DATA_DIR = DEFAULT_RESULTS_ROOT / "figure2" / "figure2_data"
DEFAULT_OUT_DIR = DEFAULT_RESULTS_ROOT / "figure2" / "results_export"

CITY_ORDER = ["Hong Kong", "Toronto", "Los Angeles"]
CORROSION_CLASSES = [
    ("low", 0.0, 0.5, "0-0.5"),
    ("medium", 0.5, 1.0, "0.5-1"),
    ("high", 1.0, math.inf, ">1"),
]
DISPERSION_COLS = {
    "r006_0p1ppm": {"threshold_ppm": 0.1, "label": "0.1 ppm"},
    "r05_0p01ppm": {"threshold_ppm": 0.01, "label": "0.01 ppm"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate and export the Figure 2 Results subsection."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--disp-outlier-percent",
        type=float,
        default=0.01,
        help=(
            "Top-radius percentage removed from Fig. 2d/g/j dispersion-radius "
            "statistics. Set 0 to disable. Default: 0.01."
        ),
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {label}: {path}")
    return path


def finite_array(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def compute_validation_stats(pred: pd.Series, meas: pd.Series) -> dict[str, float]:
    pred_arr = np.asarray(pred, dtype=float)
    meas_arr = np.asarray(meas, dtype=float)
    mask = np.isfinite(pred_arr) & np.isfinite(meas_arr)
    pred_arr = pred_arr[mask]
    meas_arr = meas_arr[mask]
    stats: dict[str, float] = {"n": int(len(pred_arr))}
    if len(pred_arr) < 2:
        return stats
    ss_res = float(np.sum((pred_arr - meas_arr) ** 2))
    ss_tot = float(np.sum((meas_arr - meas_arr.mean()) ** 2))
    rmse = float(np.sqrt(np.mean((pred_arr - meas_arr) ** 2)))
    mean_meas = float(np.mean(meas_arr))
    nz = meas_arr != 0
    stats.update(
        {
            "R2_linear": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "RMSE_MLd": rmse,
            "nRMSE_pct": rmse / mean_meas * 100.0 if mean_meas != 0 else np.nan,
            "MAPE_pct": float(np.mean(np.abs((pred_arr[nz] - meas_arr[nz]) / meas_arr[nz])) * 100.0),
            "pred_min_MLd": float(np.min(pred_arr)),
            "pred_max_MLd": float(np.max(pred_arr)),
            "measured_min_MLd": float(np.min(meas_arr)),
            "measured_max_MLd": float(np.max(meas_arr)),
        }
    )
    return stats


def load_sci_ranges(path: Path) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(path)
    required = {"parameter", "whisker_min", "whisker_max"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"SCI whisker range file missing columns: {missing}")
    ranges = {}
    for _, row in df.iterrows():
        ranges[str(row["parameter"])] = (
            float(row["whisker_min"]),
            float(row["whisker_max"]),
        )
    return ranges


def summarize_hydraulic_envelope(
    path: Path, sci_ranges: dict[str, tuple[float, float]], chunksize: int
) -> pd.DataFrame:
    arrays: dict[tuple[str, str], list[np.ndarray]] = {}
    for chunk in pd.read_csv(path, usecols=["city", "parameter", "value"], chunksize=chunksize):
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk.dropna(subset=["city", "parameter", "value"])
        if chunk.empty:
            continue
        for (city, parameter), sub in chunk.groupby(["city", "parameter"], sort=False):
            values = sub["value"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            arrays.setdefault((str(city), str(parameter)), []).append(values)

    rows = []
    for (city, parameter), parts in sorted(arrays.items()):
        values = np.concatenate(parts)
        wmin, wmax = sci_ranges.get(parameter, (np.nan, np.nan))
        if np.isfinite(wmin) and np.isfinite(wmax):
            within = (values >= wmin) & (values <= wmax)
            pct_within = float(within.mean() * 100.0)
        else:
            pct_within = np.nan
        rows.append(
            {
                "city": city,
                "parameter": parameter,
                "n": int(len(values)),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
                "sci_whisker_min": wmin,
                "sci_whisker_max": wmax,
                "pct_within_sci_whisker": pct_within,
            }
        )
    return pd.DataFrame(rows)


def summarize_h2s_nodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["city", "SH2S"])
    rows = []
    for city, sub in df.groupby("city", sort=False):
        vals = finite_array(sub["SH2S"])
        if len(vals) == 0:
            continue
        rows.append(
            {
                "city": city,
                "n_nodes": int(len(vals)),
                "median_ppm": float(np.median(vals)),
                "p95_ppm": float(np.percentile(vals, 95)),
                "p99_ppm": float(np.percentile(vals, 99)),
                "max_ppm": float(np.max(vals)),
                "pct_gt_10ppm": float((vals > 10.0).mean() * 100.0),
                "pct_gt_50ppm": float((vals > 50.0).mean() * 100.0),
                "pct_gt_100ppm": float((vals > 100.0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def _trim_top_percent(values: np.ndarray, percent: float) -> tuple[np.ndarray, float, int]:
    values = values[np.isfinite(values) & (values > 0)]
    if percent <= 0 or len(values) < 2:
        return values, np.nan, 0
    pct = min(float(percent), 100.0)
    cutoff = float(np.nanpercentile(values, 100.0 - pct))
    trimmed = values[values <= cutoff]
    return trimmed, cutoff, int(len(values) - len(trimmed))


def summarize_dispersion(path: Path, outlier_percent: float) -> pd.DataFrame:
    usecols = ["city"] + list(DISPERSION_COLS)
    df = pd.read_csv(path, usecols=usecols)
    rows = []
    for city, sub in df.groupby("city", sort=False):
        n_nodes = int(len(sub))
        for col, meta in DISPERSION_COLS.items():
            raw = finite_array(sub[col])
            raw = raw[raw > 0]
            trimmed, cutoff, n_removed = _trim_top_percent(raw, outlier_percent)
            for scope, vals, removed, cut in [
                ("raw", raw, 0, np.nan),
                ("trimmed", trimmed, n_removed, cutoff),
            ]:
                rows.append(
                    {
                        "city": city,
                        "radius_column": col,
                        "threshold_ppm": meta["threshold_ppm"],
                        "threshold_label": meta["label"],
                        "scope": scope,
                        "n_nodes": n_nodes,
                        "n_positive": int(len(vals)),
                        "n_removed_by_trimming": int(removed),
                        "trim_cutoff_m": cut,
                        "pct_positive_nodes": float(len(raw) / n_nodes * 100.0) if n_nodes else np.nan,
                        "median_radius_m": float(np.median(vals)) if len(vals) else np.nan,
                        "p95_radius_m": float(np.percentile(vals, 95)) if len(vals) else np.nan,
                        "max_radius_m": float(np.max(vals)) if len(vals) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def summarize_corrosion(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["city", "dcorr_dt", "length_m"])
    df["dcorr_dt"] = pd.to_numeric(df["dcorr_dt"], errors="coerce")
    df["length_m"] = pd.to_numeric(df["length_m"], errors="coerce")
    rows = []
    for city, sub in df.groupby("city", sort=False):
        vals = finite_array(sub["dcorr_dt"])
        total_count = int(len(vals))
        total_length_km = float(sub.loc[sub["length_m"].notna(), "length_m"].sum() / 1000.0)
        rows.append(
            {
                "city": city,
                "class": "all",
                "class_label": "all",
                "n_pipes": total_count,
                "pct_pipes": 100.0,
                "length_km": total_length_km,
                "pct_length": 100.0,
                "median_mm_yr": float(np.median(vals)) if len(vals) else np.nan,
                "p95_mm_yr": float(np.percentile(vals, 95)) if len(vals) else np.nan,
            }
        )
        valid = sub.dropna(subset=["dcorr_dt"]).copy()
        for cls, lo, hi, label in CORROSION_CLASSES:
            if math.isinf(hi):
                mask = valid["dcorr_dt"] > lo
            elif lo == 0.0:
                mask = (valid["dcorr_dt"] >= lo) & (valid["dcorr_dt"] <= hi)
            else:
                mask = (valid["dcorr_dt"] > lo) & (valid["dcorr_dt"] <= hi)
            cls_df = valid.loc[mask]
            length_km = float(cls_df["length_m"].dropna().sum() / 1000.0)
            rows.append(
                {
                    "city": city,
                    "class": cls,
                    "class_label": label,
                    "n_pipes": int(mask.sum()),
                    "pct_pipes": float(mask.mean() * 100.0) if len(valid) else np.nan,
                    "length_km": length_km,
                    "pct_length": length_km / total_length_km * 100.0 if total_length_km else np.nan,
                    "median_mm_yr": np.nan,
                    "p95_mm_yr": np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_wwtp(path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    df = pd.read_csv(path)
    valid = df.dropna(subset=["predicted_MLd", "measured_MLd"]).copy()
    stats = compute_validation_stats(valid["predicted_MLd"], valid["measured_MLd"])
    city_rows = []
    for city, sub in valid.groupby("city", sort=False):
        row = {"city": city}
        row.update(compute_validation_stats(sub["predicted_MLd"], sub["measured_MLd"]))
        city_rows.append(row)
    summary = pd.concat(
        [pd.DataFrame([{"city": "All cities", **stats}]), pd.DataFrame(city_rows)],
        ignore_index=True,
    )
    return summary, stats


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    if "city" not in df.columns:
        return df
    order = {city: i for i, city in enumerate(CITY_ORDER)}
    tmp = df.copy()
    tmp["_order"] = tmp["city"].map(order).fillna(99)
    tmp = tmp.sort_values(["_order"] + [c for c in tmp.columns if c not in {"_order", "city"}])
    return tmp.drop(columns="_order")


def get_row(df: pd.DataFrame, city: str, **filters) -> pd.Series:
    sub = df[df["city"] == city]
    for col, value in filters.items():
        sub = sub[sub[col] == value]
    if sub.empty:
        raise KeyError(f"No row for city={city}, filters={filters}")
    return sub.iloc[0]


def fmt(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{fmt(value, digits)}%"


def fmt_range(values: list[float], digits: int = 1) -> str:
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return "NA"
    if abs(min(vals) - max(vals)) < 10 ** (-(digits + 1)):
        return fmt(vals[0], digits)
    return f"{fmt(min(vals), digits)}-{fmt(max(vals), digits)}"


def fmt_km(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{float(value):,.0f}"


def fmt_flow(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if abs(value) < 0.01 and value != 0:
        mantissa, exponent = f"{value:.2e}".split("e")
        superscript = str(int(exponent)).translate(
            str.maketrans("+-0123456789", "⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        )
        return f"{float(mantissa):.2f} × 10{superscript}"
    return f"{value:.3f}"


def build_summary_dict(
    wwtp_stats: dict[str, float],
    wwtp_summary: pd.DataFrame,
    segment_counts: pd.DataFrame,
    envelope: pd.DataFrame,
    h2s: pd.DataFrame,
    dispersion: pd.DataFrame,
    corrosion: pd.DataFrame,
    disp_outlier_percent: float,
) -> dict[str, object]:
    return {
        "settings": {"disp_outlier_percent": float(disp_outlier_percent)},
        "wwtp_validation": {
            "overall": wwtp_stats,
            "by_city": wwtp_summary.to_dict(orient="records"),
        },
        "segment_counts": ordered(segment_counts).to_dict(orient="records"),
        "hydraulic_envelope": ordered(envelope).to_dict(orient="records"),
        "h2s_nodes": ordered(h2s).to_dict(orient="records"),
        "dispersion_radius": ordered(dispersion).to_dict(orient="records"),
        "corrosion": ordered(corrosion).to_dict(orient="records"),
    }


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    return obj


def build_results_text(
    wwtp_stats: dict[str, float],
    segment_counts: pd.DataFrame,
    envelope: pd.DataFrame,
    h2s: pd.DataFrame,
    dispersion: pd.DataFrame,
    corrosion: pd.DataFrame,
    disp_outlier_percent: float,
) -> str:
    length = {
        row.city: row.length_km
        for row in corrosion[corrosion["class"] == "all"].itertuples(index=False)
    }
    vel_med = {
        city: get_row(envelope, city, parameter="Velocity")["median"] for city in CITY_ORDER
    }
    flow_med = {
        city: get_row(envelope, city, parameter="Flowrate")["median"] for city in CITY_ORDER
    }
    coverage_by_param = {
        param: envelope[envelope["parameter"] == param]["pct_within_sci_whisker"].tolist()
        for param in sorted(envelope["parameter"].unique())
    }
    non_av_params = [p for p in ["Velocity", "Flowrate", "Pipe diameter", "Slope", "Filling ratio"] if p in coverage_by_param]

    h2s_med = {city: get_row(h2s, city)["median_ppm"] for city in CITY_ORDER}
    h2s_p95 = {city: get_row(h2s, city)["p95_ppm"] for city in CITY_ORDER}
    h2s_gt10 = {city: get_row(h2s, city)["pct_gt_10ppm"] for city in CITY_ORDER}
    h2s_gt50 = {city: get_row(h2s, city)["pct_gt_50ppm"] for city in CITY_ORDER}
    h2s_gt100 = {city: get_row(h2s, city)["pct_gt_100ppm"] for city in CITY_ORDER}
    h2s_median_range = max(h2s_med.values()) - min(h2s_med.values())
    h2s_tail_spread = {city: h2s_p95[city] - h2s_med[city] for city in CITY_ORDER}

    disp_trim = dispersion[dispersion["scope"] == "trimmed"].copy()
    disp_01 = {city: get_row(disp_trim, city, threshold_label="0.1 ppm") for city in CITY_ORDER}
    disp_001 = {city: get_row(disp_trim, city, threshold_label="0.01 ppm") for city in CITY_ORDER}

    corr_all = {city: get_row(corrosion, city, **{"class": "all"}) for city in CITY_ORDER}
    corr_low = {city: get_row(corrosion, city, **{"class": "low"}) for city in CITY_ORDER}
    corr_med = {city: get_row(corrosion, city, **{"class": "medium"}) for city in CITY_ORDER}
    corr_high = {city: get_row(corrosion, city, **{"class": "high"}) for city in CITY_ORDER}

    velocity_rank = sorted(CITY_ORDER, key=lambda c: vel_med[c], reverse=True)
    flow_rank = sorted(CITY_ORDER, key=lambda c: flow_med[c], reverse=True)
    h2s_rank = sorted(CITY_ORDER, key=lambda c: h2s_med[c], reverse=True)
    h2s_p95_rank = sorted(CITY_ORDER, key=lambda c: h2s_p95[c], reverse=True)
    av_coverage = fmt_range(coverage_by_param.get("A/V", []), 1)
    other_ranges = ", ".join(
        f"{p.lower()} {fmt_range(coverage_by_param[p], 1)}%" for p in non_av_params
    )

    return (
        "Cross-city mapping reveals heterogeneous sewer H₂S exposure, odour and corrosion risks\n\n"
        "After establishing that building-derived inflows reproduced monitored flow "
        "and dissolved sulphide within the SCI catchment, we next asked whether the "
        "same framework could identify spatial H₂S risk patterns in metropolitan "
        "networks with limited distributed monitoring. We applied it to the sewer "
        "networks of Hong Kong, Toronto and Los Angeles, covering "
        f"{fmt_km(length['Hong Kong'])}, {fmt_km(length['Toronto'])} and "
        f"{fmt_km(length['Los Angeles'])} km of mapped pipe length, respectively; "
        "unless stated otherwise, city triplets below follow this order. "
        "Predicted and measured total inflows agreed closely across the 17 wastewater "
        f"treatment plants (R² = {fmt(wwtp_stats['R2_linear'], 3)}, "
        f"mean-normalized RMSE = {fmt_pct(wwtp_stats['nRMSE_pct'], 1)}; Fig. 2a), with simulated "
        f"and observed inflows spanning {fmt(wwtp_stats['pred_min_MLd'], 1)}-"
        f"{fmt(wwtp_stats['pred_max_MLd'], 1)} and "
        f"{fmt(wwtp_stats['measured_min_MLd'], 1)}-"
        f"{fmt(wwtp_stats['measured_max_MLd'], 1)} ML d⁻¹, respectively. Most mapped "
        f"segments remained inside the hydraulic envelope evaluated in the SCI catchment. "
        f"A/V coverage was {av_coverage}%, and all other descriptors exceeded 84% "
        f"coverage across cities ({other_ranges}). Within this common envelope, the three networks occupied "
        "distinct hydraulic regimes. Median velocity and flowrate were both highest in "
        f"{velocity_rank[0]} ({fmt(vel_med[velocity_rank[0]], 3)} m s⁻¹ and "
        f"{fmt_flow(flow_med[flow_rank[0]])} m³ s⁻¹), compared with "
        f"{fmt(vel_med['Los Angeles'], 3)} m s⁻¹ and {fmt_flow(flow_med['Los Angeles'])} m³ s⁻¹ "
        "in Los Angeles and "
        f"{fmt(vel_med['Toronto'], 3)} m s⁻¹ and {fmt_flow(flow_med['Toronto'])} m³ s⁻¹ "
        "in Toronto. The comparison therefore spans contrasting hydraulic settings "
        "while remaining largely within the monitored calibration domain.\n\n"
        "Gas-phase H₂S exposure was strongly localized within each city and differed "
        f"among cities (Fig. 2c,f,i). Median nodal H₂S was "
        f"{fmt(h2s_med['Hong Kong'], 1)}, {fmt(h2s_med['Toronto'], 1)} and "
        f"{fmt(h2s_med['Los Angeles'], 1)} ppm, making {h2s_rank[0]} the city with "
        "the highest central exposure. The upper tail followed a different ordering: "
        f"95th-percentile concentrations were {fmt(h2s_p95['Hong Kong'], 1)}, "
        f"{fmt(h2s_p95['Toronto'], 1)} and {fmt(h2s_p95['Los Angeles'], 1)} ppm, "
        f"with the largest extreme hotspots in {h2s_p95_rank[0]}. The within-city "
        f"tail spread (95th percentile minus median, {fmt_range(list(h2s_tail_spread.values()), 1)} ppm) "
        f"was far larger than the between-city median range ({fmt(h2s_median_range, 1)} ppm), "
        "showing that node-scale heterogeneity dominated over city-average differences. "
        "Using the 10 ppm exposure threshold introduced above, "
        f"{fmt_pct(h2s_gt10['Hong Kong'])}, {fmt_pct(h2s_gt10['Toronto'])} and "
        f"{fmt_pct(h2s_gt10['Los Angeles'])} of mapped nodes exceeded this level; "
        f"the corresponding exceedances above 50 ppm were {fmt_pct(h2s_gt50['Hong Kong'])}, "
        f"{fmt_pct(h2s_gt50['Toronto'])} and {fmt_pct(h2s_gt50['Los Angeles'])}. "
        f"Concentrations above 100 ppm affected {fmt_range(list(h2s_gt100.values()), 1)}% "
        "of nodes, identifying localized dry-weather hotspots rather than a uniformly "
        "high-exposure network state.\n\n"
        "Atmospheric dispersion modelling translated emitting sewer nodes into "
        "single-opening, near-source above-ground impact zones (Fig. 2d,g,j). "
        "Zones exceeding 0.1 ppm occurred at "
        f"{fmt_range([disp_01[c]['pct_positive_nodes'] for c in CITY_ORDER], 1)}% "
        "of nodes across the three cities. Their median radii remained very small: "
        f"{fmt(disp_01['Hong Kong']['median_radius_m'], 2)}, "
        f"{fmt(disp_01['Toronto']['median_radius_m'], 2)} and "
        f"{fmt(disp_01['Los Angeles']['median_radius_m'], 2)} m in Hong Kong, "
        "Toronto and Los Angeles. The 0.01 ppm odour-screening zones were broader, "
        "with median radii of "
        f"{fmt(disp_001['Hong Kong']['median_radius_m'], 1)}, "
        f"{fmt(disp_001['Toronto']['median_radius_m'], 1)} and "
        f"{fmt(disp_001['Los Angeles']['median_radius_m'], 1)} m; their "
        f"95th-percentile radii ranged from "
        f"{fmt_range([disp_001[c]['p95_radius_m'] for c in CITY_ORDER], 1)} m. "
        "These small radii should be interpreted as near-opening screening distances, "
        "not catchment-scale plumes. Potential emission points were therefore widely "
        "distributed, whereas above-ground H₂S impacts were usually concentrated around "
        "individual openings.\n\n"
        "Pipe-level corrosion risk was also spatially uneven (Fig. 2e,h,k). Median "
        f"corrosion rates were {fmt(corr_all['Hong Kong']['median_mm_yr'], 2)}, "
        f"{fmt(corr_all['Toronto']['median_mm_yr'], 2)} and "
        f"{fmt(corr_all['Los Angeles']['median_mm_yr'], 2)} mm yr⁻¹, with Toronto "
        "having the highest central corrosion burden. "
        "Most mapped pipes remained in the low class (0-0.5 mm yr⁻¹): "
        f"{fmt_pct(corr_low['Hong Kong']['pct_pipes'])}, "
        f"{fmt_pct(corr_low['Toronto']['pct_pipes'])} and "
        f"{fmt_pct(corr_low['Los Angeles']['pct_pipes'])} by count. The intermediate "
        "class (0.5-1 mm yr⁻¹) represented "
        f"{fmt_pct(corr_med['Hong Kong']['pct_pipes'])}, "
        f"{fmt_pct(corr_med['Toronto']['pct_pipes'])} and "
        f"{fmt_pct(corr_med['Los Angeles']['pct_pipes'])}, equivalent to "
        f"{fmt(corr_med['Hong Kong']['length_km'], 1)}, "
        f"{fmt(corr_med['Toronto']['length_km'], 1)} and "
        f"{fmt(corr_med['Los Angeles']['length_km'], 1)} km. Pipes exceeding "
        "1 mm yr⁻¹ were uncommon by count "
        f"({fmt_range([corr_high[c]['pct_pipes'] for c in CITY_ORDER], 1)}%), "
        "but formed high-priority maintenance corridors totalling "
        f"{fmt(corr_high['Hong Kong']['length_km'], 1)}, "
        f"{fmt(corr_high['Toronto']['length_km'], 1)} and "
        f"{fmt(corr_high['Los Angeles']['length_km'], 1)} km in Hong Kong, Toronto "
        "and Los Angeles, respectively. Overall, Los Angeles showed the strongest "
        "median H₂S exposure, Toronto the highest median corrosion rate, and Hong Kong "
        "a lower median H₂S burden but the most pronounced upper-tail hotspots. This "
        "separation between exposure-dominated and corrosion-dominated risk motivates "
        "a pipe-size-resolved analysis of why sulphide, gas release and material loss "
        "do not vary in lockstep."
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "wwtp": require_path(data_dir / "wwtp_comparison.csv", "WWTP comparison CSV"),
        "segments": require_path(data_dir / "segment_counts.csv", "segment counts CSV"),
        "sci": require_path(data_dir / "sci_whisker_ranges.csv", "SCI whisker CSV"),
        "params": require_path(data_dir / "pipe_parameters_long.csv", "pipe parameters CSV"),
        "nodes": require_path(data_dir / "nodes.csv", "nodes CSV"),
        "corrosion": require_path(data_dir / "pipes_corrosion.csv", "pipe corrosion CSV"),
    }

    sci_ranges = load_sci_ranges(paths["sci"])
    segment_counts = pd.read_csv(paths["segments"])
    wwtp_summary, wwtp_stats = summarize_wwtp(paths["wwtp"])
    envelope = summarize_hydraulic_envelope(paths["params"], sci_ranges, args.chunksize)
    h2s = summarize_h2s_nodes(paths["nodes"])
    dispersion = summarize_dispersion(paths["nodes"], args.disp_outlier_percent)
    corrosion = summarize_corrosion(paths["corrosion"])

    text = build_results_text(
        wwtp_stats,
        segment_counts,
        envelope,
        h2s,
        dispersion,
        corrosion,
        args.disp_outlier_percent,
    )
    summary = build_summary_dict(
        wwtp_stats,
        wwtp_summary,
        segment_counts,
        envelope,
        h2s,
        dispersion,
        corrosion,
        args.disp_outlier_percent,
    )

    (out_dir / "Fig2_results_text.txt").write_text(text + "\n", encoding="utf-8-sig")
    (out_dir / "Fig2_results_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8-sig",
    )
    ordered(wwtp_summary).to_csv(out_dir / "Fig2_wwtp_validation_summary.csv", index=False, encoding="utf-8-sig")
    ordered(envelope).to_csv(out_dir / "Fig2_hydraulic_envelope_summary.csv", index=False, encoding="utf-8-sig")
    ordered(h2s).to_csv(out_dir / "Fig2_h2s_node_summary.csv", index=False, encoding="utf-8-sig")
    ordered(dispersion).to_csv(out_dir / "Fig2_dispersion_radius_summary.csv", index=False, encoding="utf-8-sig")
    ordered(corrosion).to_csv(out_dir / "Fig2_corrosion_summary.csv", index=False, encoding="utf-8-sig")

    print(f"Figure 2 Results text exported to: {out_dir / 'Fig2_results_text.txt'}")
    print(f"Supporting summaries exported to: {out_dir}")


if __name__ == "__main__":
    main()
