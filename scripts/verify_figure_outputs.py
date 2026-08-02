#!/usr/bin/env python3
"""Verify that Figures 1--5 are PNG-only and tagged at 600 dpi."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    expected = {
        "figure1": "Fig1_combined.png",
        "figure2": "Figure2_merged_params_wwtp_maps.png",
        "figure3": "Figure3.png",
    }
    failures = []
    images = []
    for directory, filename in expected.items():
        path = root / directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty: {path}")
        else:
            images.append(path)

    for directory in ("figure4", "figure5"):
        matches = sorted((root / directory).glob("*.png"))
        if len(matches) != 1:
            failures.append(
                f"{directory}: expected exactly one PNG, found {len(matches)}"
            )
        images.extend(matches)

    figure2_data = root / "figure2" / "figure2_data"
    for filename in (
        "wwtp_comparison.csv",
        "segment_counts.csv",
        "sci_whisker_ranges.csv",
        "pipe_parameters_long.csv",
        "nodes.csv",
        "pipes_corrosion.csv",
    ):
        path = figure2_data / filename
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty Figure 2 audit data: {path}")

    forbidden = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".svg"}
    ]
    failures.extend(f"forbidden image format: {path}" for path in forbidden)

    for path in images:
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    failures.append(f"not a PNG file: {path}")
                dpi = image.info.get("dpi")
                if dpi is None or any(abs(float(value) - 600.0) > 1.0 for value in dpi):
                    failures.append(f"DPI is not 600: {path} ({dpi})")
        except Exception as exc:
            failures.append(f"cannot inspect {path}: {exc}")

    if failures:
        print("Figure output verification: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("Figure output verification: PASS")
    for path in images:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
