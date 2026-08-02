#!/usr/bin/env python3
"""Launch the 81 city/scenario jobs with bounded parallelism and live progress."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from hrsnm_scenarios import scenario_axes


CITY_SPECS = {
    "hk": ("HK_v3", "hk"),
    "toronto": ("toronto_v3", "toronto"),
    "la": ("LA_v3", "la"),
}


def build_jobs(scenario_grid: str) -> list[tuple[str, str]]:
    temperatures, sulphates, cods = scenario_axes(scenario_grid)
    return [
        (city, f"T_{temp}_SO4_{so4}_COD_{cod}")
        for city in CITY_SPECS
        for temp in temperatures
        for so4 in sulphates
        for cod in cods
    ]


def expected_files(out_dir: Path, prefix: str, tag: str) -> list[Path]:
    return [
        out_dir / f"{prefix}_result_segments_v7_{tag}.csv",
        out_dir / f"{prefix}_emission_radius_segments_v7_{tag}.csv",
        out_dir / f"{prefix}_emission_radius_odour_segments_v7_{tag}.csv",
    ]


def run_job(
        root: Path, results: Path, logs: Path, city: str, tag: str,
        scenario_grid: str):
    city_dir, prefix = CITY_SPECS[city]
    out_dir = results / city_dir / "biochemical_results"
    outputs = expected_files(out_dir, prefix, tag)
    if all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        return city, tag, "SKIP", 0.0, ""

    log_path = logs / f"scenario_{city}_{tag}.log"
    command = [
        sys.executable,
        "-u",
        str(root / "run_single_scenario.py"),
        "--city", city,
        "--tag", tag,
        "--out-dir", str(out_dir),
        "--scenario-grid", scenario_grid,
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    elapsed = time.time() - started
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        return city, tag, "FAIL", elapsed, "\n".join(tail)
    if not all(path.is_file() and path.stat().st_size > 0 for path in outputs):
        return city, tag, "FAIL", elapsed, "Process exited zero but outputs are incomplete."
    return city, tag, "DONE", elapsed, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=81)
    parser.add_argument(
        "--scenario-grid", choices=("small8", "full27"), default="full27"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    results = args.results_root.resolve()
    logs = results / "logs" / "scenarios"
    logs.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.scenario_grid)
    workers = max(1, min(args.workers, len(jobs), os.cpu_count() or 1))
    print(
        f"Launching {len(jobs)} city-scenario jobs with {workers} workers. ",
        f"Progress is printed as each job finishes.",
        flush=True,
    )
    started = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_job, root, results, logs, city, tag, args.scenario_grid
            ): (city, tag)
            for city, tag in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            city, tag, status, elapsed, detail = future.result()
            print(
                f"[{index:02d}/{len(jobs)}] {status:4s} {city:7s} {tag} "
                f"({elapsed / 60:.1f} min)",
                flush=True,
            )
            if status == "FAIL":
                failures.append((city, tag, detail))
                print(detail, flush=True)

    if failures:
        raise SystemExit(f"{len(failures)} scenario job(s) failed; inspect {logs}")
    for city, (city_dir, _) in CITY_SPECS.items():
        out_dir = results / city_dir / "biochemical_results"
        expected_tags = [tag for task_city, tag in jobs if task_city == city]
        summaries = [out_dir / f"scenario_summary_v7_{tag}.csv" for tag in expected_tags]
        missing_summaries = [path for path in summaries if not path.is_file()]
        if missing_summaries:
            raise SystemExit(
                f"{city}: missing {len(missing_summaries)} per-scenario summaries"
            )
        pd.concat(
            [pd.read_csv(path, encoding="utf-8-sig") for path in summaries],
            ignore_index=True,
        ).sort_values(["temp_c", "so4_mean", "cod_mean"]).to_csv(
            out_dir / "scenario_summary_v7.csv",
            index=False,
            encoding="utf-8-sig",
        )
    print(f"All scenario jobs complete in {(time.time() - started) / 60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
