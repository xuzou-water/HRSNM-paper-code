# -*- coding: utf-8 -*-
"""
Export the Fig. 1 Results paragraph and supporting metrics.

This script follows the data choices used in:
  - figures/figure_1.py
  - hrsnm/hong_kong.py

Outputs:
  - Fig1_results_text.txt
  - Fig1_results_summary.json
  - Fig1_validation_points.csv
  - Fig1_whisker_ranges.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT.parent / f"{REPO_ROOT.name}_results"

DEFAULT_RESULT_CSV = (
    DEFAULT_RESULTS_ROOT
    / "HK_v3"
    / "biochemical_results"
    / "hk_result_segments_v7.csv"
)
DEFAULT_SEWER_CSV = Path(os.environ.get(
    "HRSNM_FIG1_SEWER_CSV",
    REPO_ROOT / "data" / "figure1" / "sewer_pipes_filled.csv",
))
DEFAULT_MEASUREMENT_CANDIDATES = [
    Path(os.environ["HRSNM_FIG1_MEASUREMENT_CSV"])
    if "HRSNM_FIG1_MEASUREMENT_CSV" in os.environ else None,
    REPO_ROOT / "data" / "figure1" / "measurement_TDS_update6.csv",
    REPO_ROOT / "data" / "figure1" / "measurement_TDS_updata6.csv",
]
DEFAULT_OUT_DIR = DEFAULT_RESULTS_ROOT / "figure1" / "results_export"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate and export the Fig. 1 Results paragraph."
    )
    parser.add_argument("--result-csv", type=Path, default=DEFAULT_RESULT_CSV)
    parser.add_argument("--sewer-csv", type=Path, default=DEFAULT_SEWER_CSV)
    parser.add_argument("--measurement-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--network-scope",
        choices=["dominant", "all"],
        default="dominant",
        help=(
            "'dominant' reproduces the SCI/SCISTW main-network filter in the "
            "Fig. 1 script; 'all' uses every simulated pipe."
        ),
    )
    return parser.parse_args()


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {label}: {path}")
    return path


def choose_measurement_path(path: Path | None) -> Path:
    if path is not None:
        return require_path(path, "measurement CSV")
    for candidate in DEFAULT_MEASUREMENT_CANDIDATES:
        if candidate is not None and candidate.exists():
            return candidate
    choices = "\n  ".join(str(p) for p in DEFAULT_MEASUREMENT_CANDIDATES if p is not None)
    raise FileNotFoundError(f"Cannot find measurement CSV. Tried:\n  {choices}")


def build_dominant_network(pipes: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Reproduce the dominant end-node filter used by the Fig. 1 script."""
    pipes_no_link = pipes[~pipes["name"].astype(str).str.startswith("Link_")].copy()
    valid = pipes_no_link["start"].notna() & pipes_no_link["end"].notna()
    valid_pipes = pipes_no_link[valid]

    forward_adj: dict[str, list[str]] = {}
    reverse_adj: dict[str, list[str]] = {}
    for start, end in zip(
        valid_pipes["start"].astype(str).values,
        valid_pipes["end"].astype(str).values,
    ):
        forward_adj.setdefault(start, []).append(end)
        reverse_adj.setdefault(end, []).append(start)
        forward_adj.setdefault(end, [])
        reverse_adj.setdefault(start, [])

    all_nodes = set(forward_adj) | set(reverse_adj)
    sink_nodes = [node for node in all_nodes if len(forward_adj.get(node, [])) == 0]

    node_to_sink: dict[str, str] = {}
    queue: deque[str] = deque()
    for sink in sink_nodes:
        node_to_sink[sink] = sink
        queue.append(sink)

    while queue:
        current = queue.popleft()
        for upstream in reverse_adj.get(current, []):
            if upstream not in node_to_sink:
                node_to_sink[upstream] = node_to_sink[current]
                queue.append(upstream)

    pipes_no_link["end_node"] = pipes_no_link["start"].astype(str).map(node_to_sink)
    if pipes_no_link["end_node"].dropna().empty:
        return pipes_no_link, None

    dominant_sink = str(pipes_no_link["end_node"].value_counts().idxmax())
    dominant = pipes_no_link[pipes_no_link["end_node"] == dominant_sink].copy()
    return dominant, dominant_sink


def tds_distribution(pipes: pd.DataFrame) -> dict[str, float]:
    tds = pd.to_numeric(pipes["SHS_out"], errors="coerce").dropna()
    if tds.empty:
        raise ValueError("No valid SHS_out values were found.")
    return {
        "n_pipes": int(len(tds)),
        "pct_below_or_equal_0_5": float((tds <= 0.5).mean() * 100.0),
        "pct_above_2": float((tds > 2.0).mean() * 100.0),
        "min": float(tds.min()),
        "max": float(tds.max()),
    }


def get_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def numeric_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def remove_outliers_iqr(data: pd.Series | np.ndarray, k: float = 1.5) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(data), errors="coerce").dropna().to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return arr
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    clean = arr[(arr >= q1 - k * iqr) & (arr <= q3 + k * iqr)]
    return clean if len(clean) else arr


def compute_whisker_ranges(
    pipes_scope: pd.DataFrame, sewer_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    sewer_df = sewer_df.copy()
    if "FR_PNT" in sewer_df.columns and "Node_Name" not in sewer_df.columns:
        sewer_df = sewer_df.rename(columns={"FR_PNT": "Node_Name"})
    if "Node_Name" not in sewer_df.columns:
        raise KeyError("sewer_pipes_filled.csv must contain Node_Name or FR_PNT.")

    pipes_scope = pipes_scope.copy()
    pipes_scope["_start_str"] = pipes_scope["start"].astype(str)
    sewer_df["Node_Name"] = sewer_df["Node_Name"].astype(str)

    pipes_dist = pd.merge(
        pipes_scope,
        sewer_df,
        left_on="_start_str",
        right_on="Node_Name",
        how="left",
        suffixes=("", "_sewer"),
    )

    specs = [
        ("A/V", ["A_V", "a_v", "AV", "av"]),
        ("Velocity", ["v", "V", "velocity", "Velocity", "VELOCITY", "v_sewer"]),
        ("Flowrate", ["flowrate", "Flowrate", "FLOWRATE", "flow", "Q"]),
        (
            "Pipe diameter",
            [
                "WIDTH",
                "width",
                "diameter",
                "DIAMETER",
                "DIAM",
                "diam",
                "width_sewer",
                "WIDTH_sewer",
                "diameter_sewer",
                "DIAMETER_sewer",
            ],
        ),
        (
            "Slope",
            ["slope", "SLOPE", "gradient", "GRADIENT", "grad", "slope_sewer"],
        ),
    ]

    records: list[dict[str, object]] = []
    ranges: dict[str, tuple[float, float]] = {}
    for label, candidates in specs:
        col = get_col(pipes_dist, candidates)
        raw = numeric_series(pipes_dist, col)
        clean = remove_outliers_iqr(raw)
        wmin = float(np.min(clean)) if len(clean) else math.nan
        wmax = float(np.max(clean)) if len(clean) else math.nan
        ranges[label] = (wmin, wmax)
        records.append(
            {
                "parameter": label,
                "source_column": col,
                "raw_n": int(len(raw)),
                "iqr_filtered_n": int(len(clean)),
                "whisker_min": wmin,
                "whisker_max": wmax,
            }
        )

    depth = pd.to_numeric(pipes_dist.get("depth"), errors="coerce")
    diameter = pd.to_numeric(pipes_dist.get("diameter"), errors="coerce")
    filling = (depth / diameter).replace([np.inf, -np.inf], np.nan)
    filling = filling[filling.notna() & (filling > 0) & (filling <= 1.5)]
    clean = remove_outliers_iqr(filling)
    wmin = float(np.min(clean)) if len(clean) else math.nan
    wmax = float(np.max(clean)) if len(clean) else math.nan
    ranges["Filling ratio"] = (wmin, wmax)
    records.append(
        {
            "parameter": "Filling ratio",
            "source_column": "depth/diameter",
            "raw_n": int(len(filling)),
            "iqr_filtered_n": int(len(clean)),
            "whisker_min": wmin,
            "whisker_max": wmax,
        }
    )

    return pd.DataFrame(records), ranges


def clean_tds_value(value: object) -> float:
    if pd.isna(value):
        return math.nan
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "na", "none"}:
        return math.nan
    for ch in ["＜", "<", "＞", ">", "≤", "≥", "≦", "≧"]:
        text = text.replace(ch, "")
    text = text.replace(" ", "").replace("，", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return math.nan


def mp_sort_key(label: str) -> int:
    match = re.search(r"\d+", str(label))
    return int(match.group()) if match else 0


def compute_validation(
    pipes: pd.DataFrame, measurement_csv: Path
) -> tuple[pd.DataFrame, dict[str, float]]:
    meas_raw = pd.read_csv(measurement_csv, sep=None, engine="python", encoding="utf-8-sig")
    if meas_raw.shape[1] < 13:
        raise ValueError("Measurement CSV does not have the expected TDS/flow columns.")

    node_row = meas_raw.iloc[0]
    meas_data = meas_raw.iloc[1:].reset_index(drop=True)
    cols = meas_raw.columns.tolist()

    tds_cols = cols[1:7]
    flow_cols = cols[-6:]
    mp_labels = [str(col).split(".")[0].strip() for col in tds_cols]
    flow_labels = [str(col).split(".")[0].strip() for col in flow_cols]
    if mp_labels != flow_labels:
        raise ValueError(
            "TDS and flowrate measurement columns are not in the same MP order: "
            f"{mp_labels} vs {flow_labels}"
        )

    for col in tds_cols:
        meas_data[col] = meas_data[col].apply(clean_tds_value)
    for col in flow_cols:
        meas_data[col] = pd.to_numeric(meas_data[col], errors="coerce")

    mp_to_node = {
        mp: str(node_row[col]).strip() for mp, col in zip(mp_labels, tds_cols)
    }
    pipes = pipes.copy()
    pipes["_start_str_lookup"] = pipes["start"].astype(str).str.strip()

    records: list[dict[str, object]] = []
    for mp, tds_col, flow_col in zip(mp_labels, tds_cols, flow_cols):
        node = mp_to_node[mp]
        match = pipes[pipes["_start_str_lookup"] == node]
        pred_tds = float(match.iloc[0]["SHS_in"]) if len(match) else math.nan
        pred_flow = float(match.iloc[0]["flowrate"] * 86400.0) if len(match) else math.nan
        meas_tds = float(meas_data[tds_col].dropna().mean())
        meas_flow = float(meas_data[flow_col].dropna().mean())
        records.append(
            {
                "MP": mp,
                "node_name": node,
                "predicted_TDS_gS_m3": pred_tds,
                "measured_TDS_gS_m3": meas_tds,
                "TDS_APE_percent": abs(pred_tds - meas_tds) / meas_tds * 100.0,
                "predicted_flow_m3_d": pred_flow,
                "measured_flow_m3_d": meas_flow,
                "predicted_flow_million_m3_d": pred_flow / 1e6,
                "measured_flow_million_m3_d": meas_flow / 1e6,
                "flow_APE_percent": abs(pred_flow - meas_flow) / meas_flow * 100.0,
            }
        )

    validation = pd.DataFrame(records).sort_values("MP", key=lambda s: s.map(mp_sort_key))
    flow_ape = validation["flow_APE_percent"].dropna().to_numpy(float)
    tds_ape = validation["TDS_APE_percent"].dropna().to_numpy(float)

    stats = {
        "n_monitoring_points": int(len(validation)),
        "flow_mare_mean_percent": float(np.mean(flow_ape)),
        "flow_mare_sd_percent": float(np.std(flow_ape, ddof=1)) if len(flow_ape) > 1 else 0.0,
        "tds_mare_mean_percent": float(np.mean(tds_ape)),
        "tds_mare_sd_percent": float(np.std(tds_ape, ddof=1)) if len(tds_ape) > 1 else 0.0,
        "sim_flow_min_million_m3_d": float(
            validation["predicted_flow_million_m3_d"].min()
        ),
        "sim_flow_max_million_m3_d": float(
            validation["predicted_flow_million_m3_d"].max()
        ),
        "obs_flow_min_million_m3_d": float(
            validation["measured_flow_million_m3_d"].min()
        ),
        "obs_flow_max_million_m3_d": float(
            validation["measured_flow_million_m3_d"].max()
        ),
        "sim_tds_min_gS_m3": float(validation["predicted_TDS_gS_m3"].min()),
        "sim_tds_max_gS_m3": float(validation["predicted_TDS_gS_m3"].max()),
        "obs_tds_min_gS_m3": float(validation["measured_TDS_gS_m3"].min()),
        "obs_tds_max_gS_m3": float(validation["measured_TDS_gS_m3"].max()),
    }
    return validation, stats


def fmt_fixed(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def fmt_trim(value: float, digits: int) -> str:
    if digits == 0:
        return str(int(round(value)))
    text = f"{value:.{digits}f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def fmt_range(low: float, high: float, digits: int, zero_threshold: float = 0.0) -> str:
    low_value = 0.0 if abs(low) <= zero_threshold else low
    return f"{fmt_trim(low_value, digits)}~{fmt_trim(high, digits)}"


def build_results_text(
    network_scope: str,
    tds_stats: dict[str, float],
    whiskers: dict[str, tuple[float, float]],
    validation_stats: dict[str, float],
) -> str:
    av = fmt_range(*whiskers["A/V"], digits=0)
    vel = fmt_range(*whiskers["Velocity"], digits=2)
    flow = fmt_range(*whiskers["Flowrate"], digits=3)
    diam = fmt_range(*whiskers["Pipe diameter"], digits=0)
    slope = fmt_range(*whiskers["Slope"], digits=3)
    filling = fmt_range(*whiskers["Filling ratio"], digits=2, zero_threshold=0.005)
    scope_phrase = "SCI network" if network_scope == "dominant" else "simulated HK network"
    n_points = validation_stats["n_monitoring_points"]
    n_points_text = "six" if n_points == 6 else str(n_points)

    return (
        f"The {scope_phrase} provided a heterogeneous hydraulic setting for "
        "evaluating pipe-resolved sulphide dynamics (Fig. 1b-h). Simulated TDS "
        f"was below 0.5 gS m-3 in {tds_stats['pct_below_or_equal_0_5']:.1f}% "
        f"of pipes, whereas {tds_stats['pct_above_2']:.1f}% exceeded 2 gS m-3, "
        "indicating localized sulphide accumulation. The hydraulic descriptors "
        "controlling sulphide formation and transport also varied widely. "
        "Excluding outliers (1.5×IQR), the whisker ranges of area-to-ventilation "
        f"ratio (A/V), velocity, flowrate, pipe diameter, slope and filling ratio "
        f"spanned {av} m-1, {vel} m s-1, {flow} m3 s-1, {diam} mm, {slope} "
        "and "
        f"{filling}, respectively. These broad ranges indicate large differences "
        "in residence time, wetted surface exposure and gas-liquid exchange "
        "potential across the monitored catchment.\n\n"
        "Despite this heterogeneity, the framework reproduced dry-weather "
        f"flowrates at the {n_points_text} monitoring "
        "points in the downstream of the SCI catchment, MP1-MP6, with a MARE of "
        f"{validation_stats['flow_mare_mean_percent']:.1f}±"
        f"{validation_stats['flow_mare_sd_percent']:.1f}% (Fig. 1i). Simulated "
        "and observed flowrates ranged from "
        f"{fmt_trim(validation_stats['sim_flow_min_million_m3_d'], 3)} to "
        f"{fmt_trim(validation_stats['sim_flow_max_million_m3_d'], 2)} and from "
        f"{fmt_trim(validation_stats['obs_flow_min_million_m3_d'], 3)} to "
        f"{fmt_trim(validation_stats['obs_flow_max_million_m3_d'], 2)} million "
        "m3 d-1, respectively. Predicted TDS concentrations were also consistent "
        "with field observations, with simulated values of "
        f"{fmt_trim(validation_stats['sim_tds_min_gS_m3'], 2)}~"
        f"{fmt_trim(validation_stats['sim_tds_max_gS_m3'], 2)} gS m-3 compared "
        "with measured values of "
        f"{fmt_trim(validation_stats['obs_tds_min_gS_m3'], 2)}~"
        f"{fmt_trim(validation_stats['obs_tds_max_gS_m3'], 2)} gS m-3, "
        "corresponding to a MARE of "
        f"{validation_stats['tds_mare_mean_percent']:.1f}±"
        f"{validation_stats['tds_mare_sd_percent']:.1f}% (Fig. 1j). The joint "
        "agreement in flow and TDS indicates that building-derived wastewater "
        "allocation captures the dominant spatial structure of dry-weather sewer "
        "inflows sufficiently to drive pipe-resolved sulphide simulations. "
        "Furthermore, it establishes an empirical envelope of conditions for "
        "applying this framework across larger, more complex urban scales."
    )


def main() -> None:
    args = parse_args()
    result_csv = require_path(args.result_csv, "HK result CSV")
    sewer_csv = require_path(args.sewer_csv, "sewer pipe CSV")
    measurement_csv = choose_measurement_path(args.measurement_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipes = pd.read_csv(result_csv)
    sewer_df = pd.read_csv(sewer_csv)

    if args.network_scope == "dominant":
        pipes_scope, dominant_sink = build_dominant_network(pipes)
    else:
        pipes_scope = pipes.copy()
        dominant_sink = None

    tds_stats = tds_distribution(pipes_scope)
    whisker_df, whiskers = compute_whisker_ranges(pipes_scope, sewer_df)
    validation_df, validation_stats = compute_validation(pipes, measurement_csv)
    results_text = build_results_text(
        args.network_scope, tds_stats, whiskers, validation_stats
    )

    summary = {
        "inputs": {
            "result_csv": str(result_csv),
            "sewer_csv": str(sewer_csv),
            "measurement_csv": str(measurement_csv),
            "network_scope": args.network_scope,
            "dominant_sink": dominant_sink,
        },
        "tds_distribution": tds_stats,
        "validation": validation_stats,
    }

    text_path = args.out_dir / "Fig1_results_text.txt"
    summary_path = args.out_dir / "Fig1_results_summary.json"
    validation_path = args.out_dir / "Fig1_validation_points.csv"
    whisker_path = args.out_dir / "Fig1_whisker_ranges.csv"

    text_path.write_text(results_text + "\n", encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8-sig"
    )
    validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
    whisker_df.to_csv(whisker_path, index=False, encoding="utf-8-sig")

    print(f"Results text: {text_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Validation CSV: {validation_path}")
    print(f"Whisker CSV: {whisker_path}")


if __name__ == "__main__":
    main()
