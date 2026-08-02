#!/usr/bin/env python3
"""Validate the code release and, optionally, its external input dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_DATA = [
    "data/processed_data/hk_v3/hydraulic_results/segment_hydraulics.csv",
    "data/processed_data/hk_v3/hydraulic_results/nodes_all.csv",
    "data/processed_data/hk_v3/hydraulic_results/hk_building_details.csv",
    "data/processed_data/hk_v3/hydraulic_nodes.csv",
    "data/processed_data/toronto_v3/hydraulic_results/segment_hydraulics.csv",
    "data/processed_data/toronto_v3/hydraulic_results/nodes_all.csv",
    "data/processed_data/toronto_v3/hydraulic_results/toronto_building_details.csv",
    "data/processed_data/toronto_v3/hydraulic_nodes.csv",
    "data/processed_data/la_v3/hydraulic_results/segment_hydraulics.csv",
    "data/processed_data/la_v3/hydraulic_results/nodes_all.csv",
    "data/processed_data/la_v3/hydraulic_results/la_building_details.csv",
    "data/processed_data/la_v3/hydraulic_nodes.csv",
    "data/global/catchment_summary.csv",
    "data/global/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp",
    "data/global/ne_10m_admin_0_countries/ne_10m_admin_0_countries.dbf",
    "data/global/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shx",
    "data/figure1/HK_buildings.shp",
    "data/figure1/HK_buildings.dbf",
    "data/figure1/sewer_pipes_filled.csv",
    "data/figure1/total_hrt_per_node.csv",
    "data/figure1/measurement_TDS_update6.csv",
    "data/figure2/17WWTP_inflow_data1.xlsx",
    "data/figure2/HK_border/hk_merged_border.shp",
    "data/figure2/HK_border/hk_merged_border.shx",
    "data/figure2/HK_border/hk_merged_border.dbf",
    "data/figure2/HK_border/hk_merged_border.prj",
    "data/figure2/HK_maintenance/Maintenance_project.shp",
    "data/figure2/HK_maintenance/Maintenance_project.shx",
    "data/figure2/HK_maintenance/Maintenance_project.dbf",
    "data/figure2/HK_maintenance/Maintenance_project.prj",
    "data/figure2/Toronto_border/citygcs_regional_mun_wgs84.shp",
    "data/figure2/Toronto_border/citygcs_regional_mun_wgs84.shx",
    "data/figure2/Toronto_border/citygcs_regional_mun_wgs84.dbf",
    "data/figure2/Toronto_border/citygcs_regional_mun_wgs84.prj",
    "data/figure2/LA_border/City_Boundary.shp",
    "data/figure2/LA_border/City_Boundary.shx",
    "data/figure2/LA_border/City_Boundary.dbf",
    "data/figure2/LA_border/City_Boundary.prj",
]
REQUIRED_CODE = [
    "hk_HRSNM_v7_5_test2.py",
    "toronto_HRSNM_v7_5_test1.py",
    "la_HRSNM_v7_5_test1.py",
    "HRSNM(v7 Fig1_4_test2).py",
    "HRSNM(v Fig2_9).py",
    "HRSNM(v7 Figure3).py",
    "HRSNM(v7 Fig4_2).py",
    "HRSNM(v7 Fig5_3).py",
    "hrsnm_dissolved_oxygen.py",
    "hrsnm_node_mixing.py",
    "hrsnm_scenarios.py",
    "corrosion_criterion.py",
    "run_full27_figures_12345.sh",
    "run_parallel_scenarios.py",
    "run_single_scenario.py",
    "export_fig1_results_text.py",
    "export_fig2_results_text.py",
    "plot_figure_s4_50year_failure.py",
    "test_corrosion_criterion.py",
    "test_hrsnm_node_mixing.py",
    "test_hrsnm_scenarios.py",
    "test_hrsnm_dissolved_oxygen.py",
    "test_figure2_inputs.py",
    "verify_figure_outputs.py",
    "environment.yml",
    "README.md",
    "README_CN.md",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument(
        "--check-data", action="store_true",
        help="also require the non-repository input files listed in data/README.md",
    )
    args = parser.parse_args()
    failures = []
    required = REQUIRED_CODE + (REQUIRED_DATA if args.check_data else [])
    for relative in required:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty: {relative}")

    expected_outfalls = {
        "data/processed_data/hk_v3/hydraulic_nodes.csv": 9,
        "data/processed_data/toronto_v3/hydraulic_nodes.csv": 5,
        "data/processed_data/la_v3/hydraulic_nodes.csv": 3,
    }
    for relative, expected in expected_outfalls.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            count = sum(
                1 for row in csv.DictReader(stream)
                if str(row.get("type", "")).strip().lower() == "outfall"
            )
        if count != expected:
            failures.append(
                f"{relative}: expected {expected} outfalls, found {count}"
            )

    model_expectations = {
        "hk_HRSNM_v7_5_test2.py": (
            "TEMP = 29.0",
            "COD_PER_SOURCE = 488.0",
            "COD_CI = 52.0",
            "CORR_VALUE = 0.8",
            "R_GAS * T_H2S_K",
            "saturation_do_mg_l(temp_c)",
        ),
        "toronto_HRSNM_v7_5_test1.py": (
            "CORR_VALUE = 0.8",
            "R_GAS * T_H2S_K",
            "saturation_do_mg_l(temp_c)",
        ),
        "la_HRSNM_v7_5_test1.py": (
            "CORR_VALUE = 0.8",
            "R_GAS * T_H2S_K",
            "saturation_do_mg_l(temp_c)",
        ),
    }
    for relative, snippets in model_expectations.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        for snippet in snippets:
            if snippet not in source:
                failures.append(f"{relative}: missing model setting {snippet!r}")

    if args.require_clean:
        forbidden_dirs = [
            path for path in ROOT.iterdir()
            if path.is_dir() and (
                path.name.endswith("_results")
                or path.name.endswith("_results_export")
                or path.name.startswith("node_dwf_fix_results")
            )
        ]
        forbidden_outputs = [
            path for path in ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg"}
        ]
        compiled_artifacts = [
            path for path in ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pyc", ".pyo"}
        ]
        generated_logs = [
            path for path in ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".log", ".pid"}
        ]
        development_files = [
            path for path in ROOT.rglob("*")
            if path.is_file() and (
                path.suffix.lower() == ".docx"
                or path.name == ".Rhistory"
                or path.name.startswith(".~lock.")
            )
        ]
        development_dirs = [
            path for path in ROOT.iterdir()
            if path.is_dir() and path.name in {
                ".agents", ".codex", "__pycache__"
            }
        ]
        failures.extend(f"generated result directory present: {p.name}" for p in forbidden_dirs)
        failures.extend(f"generated image present: {p.relative_to(ROOT)}" for p in forbidden_outputs)
        failures.extend(f"compiled artifact present: {p.relative_to(ROOT)}" for p in compiled_artifacts)
        failures.extend(f"generated log/PID present: {p.relative_to(ROOT)}" for p in generated_logs)
        failures.extend(f"development file present: {p.relative_to(ROOT)}" for p in development_files)
        failures.extend(f"development directory present: {p.relative_to(ROOT)}" for p in development_dirs)

    if failures:
        print("Package validation: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(1)
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    print(f"Package validation: PASS ({len(files)} files, {total / 1024**3:.3f} GiB)")


if __name__ == "__main__":
    main()
