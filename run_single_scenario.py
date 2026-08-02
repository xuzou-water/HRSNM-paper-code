#!/usr/bin/env python3
"""Run exactly one city/scenario without repeating the baseline calculation."""

from __future__ import annotations

import argparse
import importlib
import os
import re
import time
from pathlib import Path

import pandas as pd

from hrsnm_scenarios import scenario_axes


TAG_RE = re.compile(r"^T_(\d+)_SO4_(\d+)_COD_(\d+)$")


def parse_tag(tag: str, scenario_grid: str = "full27") -> tuple[int, int, int]:
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError(f"Invalid scenario tag: {tag!r}")
    values = tuple(int(value) for value in match.groups())
    axes = scenario_axes(scenario_grid)
    if any(value not in axis for value, axis in zip(values, axes)):
        raise ValueError(f"Scenario is outside the {scenario_grid} grid: {tag}")
    return values


def prepare_city(city: str, out_dir: Path):
    if city == "hk":
        module = importlib.import_module("hk_HRSNM_v7_5_test2")
        raw, nodes = module.load_segments_from_code2()
        pipes = module.attach_node_is_original(module.compute_geometry(raw), nodes)
    elif city == "toronto":
        module = importlib.import_module("toronto_HRSNM_v7_5_test1")
        raw, nodes = module.load_segments()
        pipes = module.compute_geometry(raw)
    elif city == "la":
        module = importlib.import_module("la_HRSNM_v7_5_test1")
        raw, nodes = module.load_segments()
        pipes = module.attach_original_flags(module.compute_geometry(raw), nodes)
    else:
        raise ValueError(f"Unknown city: {city}")

    module.OUT_DIR = str(out_dir.resolve())
    topo = module.build_topology(pipes)
    node_dwf, positive_nodes = module.build_node_dwf(nodes, topo[0].nodes())
    return module, pipes, nodes, topo, node_dwf, positive_nodes


def run(city: str, tag: str, out_dir: Path, scenario_grid: str = "full27") -> None:
    temp, so4, cod = parse_tag(tag, scenario_grid)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"[{city}/{tag}] loading hydraulic network", flush=True)
    module, pipes, nodes, topo, node_dwf, positive_nodes = prepare_city(city, out_dir)
    so4_ci = so4 * module.SO4_CV_BASELINE
    cod_ci = cod * module.COD_CV_BASELINE

    print(f"[{city}/{tag}] simulating", flush=True)
    if city == "hk":
        conc_in, conc_out, _ = module.run_simulation_hk(
            pipes, topo, node_dwf, positive_nodes,
            so4_mean=so4, cod_mean=cod, so4_ci=so4_ci,
            cod_ci=cod_ci, temp_c=temp,
        )
        result = module.postprocess_and_save_hk(
            pipes, conc_in, conc_out, tag=tag, temp_c=temp
        )
        emission = module.compute_h2s_emission_hk(result, tag=tag, temp_c=temp)
    elif city == "toronto":
        conc_in, conc_out, _ = module.run_simulation_toronto(
            pipes, topo, node_dwf, positive_nodes,
            so4_mean=so4, cod_mean=cod, so4_ci=so4_ci,
            cod_ci=cod_ci, temp_c=temp,
        )
        result = module.postprocess_and_save_toronto(
            pipes, conc_in, conc_out, tag=tag, temp_c=temp
        )
        emission = module.compute_h2s_emission_toronto(
            result, tag=tag, temp_c=temp, nodes_df=nodes
        )
    else:
        conc_in, conc_out, _ = module.run_simulation_la(
            pipes, topo, node_dwf, positive_nodes,
            so4_mean=so4, cod_mean=cod, so4_ci=so4_ci,
            cod_ci=cod_ci, temp_c=temp,
        )
        result = module.postprocess_and_save_la(
            pipes, conc_in, conc_out, tag=tag, temp_c=temp
        )
        emission = module.compute_h2s_emission_la(result, tag=tag, temp_c=temp)

    summary = pd.DataFrame([{
        "tag": tag,
        "temp_c": temp,
        "so4_mean": so4,
        "so4_ci": so4_ci,
        "cod_mean": cod,
        "cod_ci": cod_ci,
        "mean_SHS_in": result["SHS_in"].mean(),
        "mean_SHS_out": result["SHS_out"].mean(),
        "max_SHS_out": result["SHS_out"].max(),
        "mean_H2S_ppm": emission["SH2S"].mean(),
        "max_H2S_ppm": emission["SH2S"].max(),
        "mean_dcorr_dt": emission["dcorr_dt"].mean(),
        "max_dcorr_dt": emission["dcorr_dt"].max(),
        "total_H2S_emission_rate": emission["H2S_emission_rate"].sum(),
    }])
    summary.to_csv(
        out_dir / f"scenario_summary_v7_{tag}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"[{city}/{tag}] complete in {(time.time() - started) / 60:.1f} min",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=("hk", "toronto", "la"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--scenario-grid", choices=("small8", "full27"), default="full27"
    )
    args = parser.parse_args()
    run(args.city, args.tag, args.out_dir, args.scenario_grid)


if __name__ == "__main__":
    main()
