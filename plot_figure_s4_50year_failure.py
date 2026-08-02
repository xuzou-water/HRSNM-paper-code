#!/usr/bin/env python3
"""Map sewer-pipe length that fails within a 50-year corrosion horizon."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from corrosion_criterion import (
    DEFAULT_CORROSION_HORIZON_YEARS,
    DEFAULT_WALL_THICKNESS_FRACTION,
    corrosion_exceeds_assumed_wall,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / f"{SCRIPT_DIR.name}_results"
DEFAULT_BORDER_ROOT = SCRIPT_DIR / "data" / "figure2"
DPI = 600

FAILED_COLOR = "#A33A2B"
NOT_FAILED_COLOR = "#8EA49A"
UNKNOWN_COLOR = "#5B5B5B"
BORDER_COLOR = "#ECE8DF"
MAINTENANCE_COLOR = "#174A7E"
FAILED_MAINTENANCE_COLOR = "#6A2C70"

CITY_CONFIGS = (
    {
        "name": "Hong Kong",
        "directory": "HK_v3",
        "prefix": "hk",
        "crs": "EPSG:2326",
        "border": Path("HK_border") / "hk_merged_border.shp",
        "border_crs": "EPSG:2326",
        "maintenance": Path("HK_maintenance") / "Maintenance_project.shp",
        "xlim": (807000, 850000),
        "ylim": (807000, 847000),
        "scalebar_m": 5000,
    },
    {
        "name": "Toronto",
        "directory": "toronto_v3",
        "prefix": "toronto",
        "crs": "EPSG:26917",
        "border": Path("Toronto_border") / "citygcs_regional_mun_wgs84.shp",
        "border_crs": "EPSG:4326",
        "xlim": (607000, 652000),
        "ylim": (4823000, 4860000),
        "scalebar_m": 5000,
    },
    {
        "name": "Los Angeles",
        "directory": "LA_v3",
        "prefix": "la",
        "crs": "EPSG:26945",
        "border": Path("LA_border") / "City_Boundary.shp",
        "border_crs": "EPSG:4326",
        "xlim": (1935000, 1990000),
        "ylim": (545000, 595000),
        "scalebar_m": 10000,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure S4 showing sewer segments predicted to fail "
            "within the corrosion design horizon, with length-weighted statistics."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(os.environ.get("RESULTS_ROOT", DEFAULT_RESULTS_ROOT)),
        help="Root containing the three baseline biochemical_results directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: RESULTS_ROOT/figureS4).",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=DEFAULT_CORROSION_HORIZON_YEARS,
        help="Corrosion horizon in years (default: 50).",
    )
    parser.add_argument(
        "--wall-fraction",
        type=float,
        default=DEFAULT_WALL_THICKNESS_FRACTION,
        help="Assumed pipe wall thickness / diameter (default: 0.075).",
    )
    parser.add_argument(
        "--border-root",
        type=Path,
        default=DEFAULT_BORDER_ROOT,
        help="Optional Figure 2 administrative-boundary input directory.",
    )
    parser.add_argument(
        "--no-borders",
        action="store_true",
        help="Draw pipe networks without administrative boundary polygons.",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure.")
    return parser.parse_args()


def require_columns(path: Path, expected: set[str]) -> None:
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = expected - columns
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")


def load_city(
    config: dict,
    results_root: Path,
    *,
    figure2_root: Path,
    years: float,
    wall_fraction: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int | float | str],
]:
    bio_dir = results_root / config["directory"] / "biochemical_results"
    result_path = bio_dir / f"{config['prefix']}_result_segments_v7.csv"
    corrosion_path = bio_dir / f"{config['prefix']}_emission_radius_segments_v7.csv"
    for path in (result_path, corrosion_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required baseline result not found: {path}")

    result_columns = {
        "name",
        "parent_link",
        "diameter",
        "length",
        "us_x",
        "us_y",
        "ds_x",
        "ds_y",
    }
    corrosion_columns = {"pipe_name", "dcorr_dt"}
    require_columns(result_path, result_columns)
    require_columns(corrosion_path, corrosion_columns)

    result = pd.read_csv(
        result_path,
        usecols=sorted(result_columns),
        dtype={"name": "string", "parent_link": "string"},
    )
    corrosion = pd.read_csv(
        corrosion_path,
        usecols=sorted(corrosion_columns),
        dtype={"pipe_name": "string"},
    )

    if (
        len(result) == len(corrosion)
        and result["name"].array.equals(corrosion["pipe_name"].array)
    ):
        result["dcorr_dt"] = pd.to_numeric(corrosion["dcorr_dt"], errors="coerce")
    else:
        result = result.merge(
            corrosion,
            left_on="name",
            right_on="pipe_name",
            how="left",
            validate="one_to_one",
        )
        if result["dcorr_dt"].isna().all():
            raise ValueError(
                f"No segment names matched between {result_path} and {corrosion_path}"
            )

    rate = pd.to_numeric(result["dcorr_dt"], errors="coerce").to_numpy(dtype=float)
    diameter = pd.to_numeric(result["diameter"], errors="coerce").to_numpy(dtype=float)
    length_m = pd.to_numeric(result["length"], errors="coerce").to_numpy(dtype=float)
    criterion_valid = (
        np.isfinite(rate)
        & np.isfinite(diameter)
        & (rate >= 0)
        & (diameter > 0)
    )
    length_valid = np.isfinite(length_m) & (length_m > 0)
    segment_assessed = criterion_valid & length_valid
    segment_failed = corrosion_exceeds_assumed_wall(
        rate,
        diameter,
        years=years,
        wall_fraction=wall_fraction,
    ) & length_valid

    coords = np.empty((len(result), 2, 2), dtype=np.float32)
    coords[:, 0, 0] = pd.to_numeric(result["us_x"], errors="coerce")
    coords[:, 0, 1] = pd.to_numeric(result["us_y"], errors="coerce")
    coords[:, 1, 0] = pd.to_numeric(result["ds_x"], errors="coerce")
    coords[:, 1, 1] = pd.to_numeric(result["ds_y"], errors="coerce")
    geometry_valid = np.isfinite(coords).all(axis=(1, 2))
    if not geometry_valid.any():
        raise ValueError(f"No valid pipe coordinates found in {result_path}")

    coords = coords[geometry_valid]
    mapped_failed = segment_failed[geometry_valid]
    mapped_assessed = segment_assessed[geometry_valid]
    segment_in_maintenance = np.zeros(len(result), dtype=bool)

    total_length_m = float(length_m[segment_assessed].sum())
    failed_length_m = float(length_m[segment_failed].sum())
    not_failed_length_m = total_length_m - failed_length_m
    stats: dict[str, int | float | str] = {
        "city": config["name"],
        "total_assessed_length_km": total_length_m / 1000.0,
        "failed_length_km": failed_length_m / 1000.0,
        "not_failed_length_km": not_failed_length_m / 1000.0,
        "failure_percentage_of_assessed_length": (
            100.0 * failed_length_m / total_length_m
            if total_length_m > 0
            else np.nan
        ),
        "computational_segments": int(len(result)),
        "failed_segments": int(segment_failed.sum()),
        "unassessed_segments": int((~segment_assessed).sum()),
        "unassessed_length_with_valid_length_km": (
            float(length_m[length_valid & ~criterion_valid].sum()) / 1000.0
        ),
        "segments_without_geometry": int((~geometry_valid).sum()),
        "result_file": str(result_path),
        "corrosion_file": str(corrosion_path),
    }

    maintenance_relative = config.get("maintenance")
    if maintenance_relative is not None:
        maintenance_path = figure2_root / maintenance_relative
        if not maintenance_path.is_file():
            raise FileNotFoundError(
                f"Hong Kong maintenance map not found: {maintenance_path}"
            )
        import geopandas as gpd

        maintenance = gpd.read_file(maintenance_path)
        if "pipe_name" not in maintenance.columns:
            raise KeyError(f"{maintenance_path} is missing required column: pipe_name")
        maintenance_names = set(
            maintenance["pipe_name"].dropna().astype(str).str.strip()
        )
        maintenance_names.discard("")
        parent_names = result["parent_link"].astype("string")
        segment_in_maintenance = parent_names.isin(maintenance_names).to_numpy()
        matched_names = set(parent_names.loc[segment_in_maintenance].dropna().astype(str))

        maintenance_assessed = segment_assessed & segment_in_maintenance
        failed_maintenance = segment_failed & segment_in_maintenance
        maintenance_length_m = float(length_m[maintenance_assessed].sum())
        failed_maintenance_length_m = float(length_m[failed_maintenance].sum())
        failed_coverage = (
            100.0 * failed_maintenance_length_m / failed_length_m
            if failed_length_m > 0
            else np.nan
        )
        maintenance_failure_rate = (
            100.0 * failed_maintenance_length_m / maintenance_length_m
            if maintenance_length_m > 0
            else np.nan
        )
        union_length_m = (
            failed_length_m + maintenance_length_m - failed_maintenance_length_m
        )
        length_jaccard = (
            100.0 * failed_maintenance_length_m / union_length_m
            if union_length_m > 0
            else np.nan
        )
        stats.update(
            {
                "maintenance_source_file": str(maintenance_path),
                "maintenance_source_features": int(len(maintenance)),
                "maintenance_unique_pipe_names": int(len(maintenance_names)),
                "maintenance_matched_pipe_names": int(len(matched_names)),
                "maintenance_unmatched_pipe_names": int(
                    len(maintenance_names - matched_names)
                ),
                "maintenance_mapped_length_km": maintenance_length_m / 1000.0,
                "failed_within_maintenance_length_km": (
                    failed_maintenance_length_m / 1000.0
                ),
                "maintenance_coverage_of_failed_length_pct": failed_coverage,
                "failure_rate_within_maintenance_pct": maintenance_failure_rate,
                "failed_maintenance_length_jaccard_pct": length_jaccard,
                "failed_segments_within_maintenance": int(
                    failed_maintenance.sum()
                ),
            }
        )

    return (
        coords,
        mapped_failed,
        mapped_assessed,
        segment_in_maintenance[geometry_valid],
        stats,
    )


def draw_border(ax, config: dict, border_root: Path) -> None:
    path = border_root / config["border"]
    if not path.is_file():
        print(f"  Optional border not found; continuing without it: {path}")
        return
    try:
        import geopandas as gpd

        border = gpd.read_file(path)
        if border.crs is None:
            border = border.set_crs(config["border_crs"])
        border = border.to_crs(config["crs"])
        border.plot(
            ax=ax,
            facecolor=BORDER_COLOR,
            edgecolor="none",
            linewidth=0,
            zorder=0,
        )
    except Exception as exc:
        print(f"  Optional border could not be drawn ({path}): {exc}")


def draw_maintenance_map(ax, config: dict, figure2_root: Path) -> None:
    relative = config.get("maintenance")
    if relative is None:
        return
    path = figure2_root / relative
    import geopandas as gpd

    maintenance = gpd.read_file(path)
    if maintenance.crs is None:
        maintenance = maintenance.set_crs(config["crs"])
    maintenance = maintenance.to_crs(config["crs"])
    maintenance.plot(
        ax=ax,
        color=MAINTENANCE_COLOR,
        linewidth=0.55,
        alpha=0.82,
        zorder=2,
    )


def add_scalebar(ax, length_m: float) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]
    x0 = xlim[0] + width * 0.07
    y0 = ylim[0] + height * 0.07
    ax.plot(
        [x0, x0 + length_m],
        [y0, y0],
        color="#222222",
        linewidth=1.3,
        solid_capstyle="butt",
        zorder=5,
    )
    tick = height * 0.012
    ax.plot([x0, x0], [y0 - tick, y0 + tick], color="#222222", linewidth=0.8)
    ax.plot(
        [x0 + length_m, x0 + length_m],
        [y0 - tick, y0 + tick],
        color="#222222",
        linewidth=0.8,
    )
    label = f"{length_m / 1000:g} km"
    ax.text(x0 + length_m / 2, y0 + height * 0.018, label, ha="center", fontsize=7)


def add_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.94, 0.91),
        xytext=(0.94, 0.80),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "#222222", "lw": 1.0},
        zorder=6,
    )


def plot_city(
    ax,
    config: dict,
    coords: np.ndarray,
    failed: np.ndarray,
    assessed: np.ndarray,
    in_maintenance: np.ndarray,
    stats: dict[str, int | float | str],
    *,
    border_root: Path,
    draw_borders: bool,
    panel_label: str,
) -> None:
    if draw_borders:
        draw_border(ax, config, border_root)

    unknown = ~assessed
    not_failed = assessed & ~failed
    if not_failed.any():
        ax.add_collection(
            LineCollection(
                coords[not_failed],
                colors=NOT_FAILED_COLOR,
                linewidths=0.10,
                alpha=0.72,
                zorder=1,
                rasterized=True,
            )
        )
    if unknown.any():
        ax.add_collection(
            LineCollection(
                coords[unknown],
                colors=UNKNOWN_COLOR,
                linewidths=0.12,
                alpha=0.75,
                zorder=2,
                rasterized=True,
            )
        )
    if config.get("maintenance") is not None:
        draw_maintenance_map(ax, config, border_root)
    if failed.any():
        ax.add_collection(
            LineCollection(
                coords[failed],
                colors=FAILED_COLOR,
                linewidths=0.22,
                alpha=0.92,
                zorder=3,
                rasterized=True,
            )
        )
    failed_maintenance = failed & in_maintenance
    if failed_maintenance.any():
        ax.add_collection(
            LineCollection(
                coords[failed_maintenance],
                colors=FAILED_MAINTENANCE_COLOR,
                linewidths=0.48,
                alpha=1.0,
                zorder=4,
                rasterized=True,
            )
        )

    ax.set_xlim(config["xlim"])
    ax.set_ylim(config["ylim"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    percentage = float(stats["failure_percentage_of_assessed_length"])
    title = (
        f"{config['name']}\n"
        f"{float(stats['failed_length_km']):,.1f} / "
        f"{float(stats['total_assessed_length_km']):,.1f} km failed "
        f"({percentage:.2f}%)"
    )
    if "maintenance_coverage_of_failed_length_pct" in stats:
        title += (
            "\n"
            f"Maintenance overlap: "
            f"{float(stats['failed_within_maintenance_length_km']):.1f} km "
            f"({float(stats['maintenance_coverage_of_failed_length_pct']):.2f}% "
            "of failed length)"
        )
    ax.set_title(title, fontsize=9.5, pad=7)
    ax.text(
        0.01,
        0.99,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        zorder=8,
    )
    add_scalebar(ax, config["scalebar_m"])
    add_north_arrow(ax)


def write_summary(
    path: Path,
    stats_rows: list[dict[str, int | float | str]],
    *,
    years: float,
    wall_fraction: float,
) -> None:
    total = sum(float(row["total_assessed_length_km"]) for row in stats_rows)
    failed = sum(float(row["failed_length_km"]) for row in stats_rows)
    not_failed = sum(float(row["not_failed_length_km"]) for row in stats_rows)
    combined_percentage = 100.0 * failed / total if total > 0 else np.nan

    lines = [
        "Figure S4: 50-year sewer-pipe corrosion failure summary",
        "=" * 62,
        "",
        "Criterion:",
        "  cumulative_corrosion_mm = corrosion_rate_mm_per_year * years",
        "  assumed_wall_thickness_mm = diameter_m * 1000 * wall_fraction",
        "  failed segment: cumulative_corrosion_mm > assumed_wall_thickness_mm",
        "  each computational segment is mapped independently",
        f"  years = {years:g}",
        f"  wall_fraction = {wall_fraction:g} ({wall_fraction * 100:g}% of diameter)",
        "  failed length percentage = sum(failed segment length) /",
        "                             sum(all assessed positive segment length) * 100",
        "",
        (
            "City\tAssessed length (km)\tFailed length (km)"
            "\tNot-failed length (km)\tFailed length (%)"
            "\tComputational segments\tFailed segments\tUnassessed segments"
        ),
    ]
    for row in stats_rows:
        lines.append(
            f"{row['city']}\t{float(row['total_assessed_length_km']):.6f}\t"
            f"{float(row['failed_length_km']):.6f}\t"
            f"{float(row['not_failed_length_km']):.6f}\t"
            f"{float(row['failure_percentage_of_assessed_length']):.6f}\t"
            f"{int(row['computational_segments'])}\t"
            f"{int(row['failed_segments'])}\t"
            f"{int(row['unassessed_segments'])}"
        )
    lines.extend(
        [
            (
                f"Three-city total\t{total:.6f}\t{failed:.6f}\t"
                f"{not_failed:.6f}\t{combined_percentage:.6f}\t"
                f"{sum(int(row['computational_segments']) for row in stats_rows)}\t"
                f"{sum(int(row['failed_segments']) for row in stats_rows)}\t"
                f"{sum(int(row['unassessed_segments']) for row in stats_rows)}"
            ),
            "",
            "Input files:",
        ]
    )
    for row in stats_rows:
        lines.append(f"  {row['city']} results: {row['result_file']}")
        lines.append(f"  {row['city']} corrosion: {row['corrosion_file']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hk_maintenance_summary(
    path: Path,
    stats: dict[str, int | float | str],
    *,
    years: float,
    wall_fraction: float,
) -> None:
    required = {
        "maintenance_source_file",
        "maintenance_mapped_length_km",
        "failed_within_maintenance_length_km",
    }
    missing = required - set(stats)
    if missing:
        raise KeyError(f"Hong Kong maintenance statistics are missing: {sorted(missing)}")

    lines = [
        "Figure S4: Hong Kong 50-year failure / maintenance overlap",
        "=" * 68,
        "",
        f"Corrosion horizon (years): {years:g}",
        f"Assumed wall-thickness fraction: {wall_fraction:g}",
        f"Maintenance source: {stats['maintenance_source_file']}",
        "",
        "Length results:",
        (
            f"  Total assessed Hong Kong pipe length (km): "
            f"{float(stats['total_assessed_length_km']):.6f}"
        ),
        (
            f"  Total predicted failed length (km): "
            f"{float(stats['failed_length_km']):.6f}"
        ),
        (
            f"  Maintenance-map pipe length matched to model (km): "
            f"{float(stats['maintenance_mapped_length_km']):.6f}"
        ),
        (
            f"  Predicted failed length within maintenance map (km): "
            f"{float(stats['failed_within_maintenance_length_km']):.6f}"
        ),
        "",
        "Overlap metrics:",
        (
            "  Maintenance coverage of predicted failed length (%): "
            f"{float(stats['maintenance_coverage_of_failed_length_pct']):.6f}"
        ),
        (
            "  Predicted failure rate within maintenance-map length (%): "
            f"{float(stats['failure_rate_within_maintenance_pct']):.6f}"
        ),
        (
            "  Length-weighted Jaccard overlap (%): "
            f"{float(stats['failed_maintenance_length_jaccard_pct']):.6f}"
        ),
        "",
        "Matching audit:",
        (
            f"  Maintenance source features: "
            f"{int(stats['maintenance_source_features'])}"
        ),
        (
            f"  Unique maintenance pipe names: "
            f"{int(stats['maintenance_unique_pipe_names'])}"
        ),
        (
            f"  Maintenance names matched to model parent_link: "
            f"{int(stats['maintenance_matched_pipe_names'])}"
        ),
        (
            f"  Maintenance names not present in model results: "
            f"{int(stats['maintenance_unmatched_pipe_names'])}"
        ),
        (
            f"  Failed computational segments within maintenance map: "
            f"{int(stats['failed_segments_within_maintenance'])}"
        ),
        "",
        "Definitions:",
        (
            "  Coverage = failed length inside maintenance map / "
            "all predicted failed length."
        ),
        (
            "  Within-maintenance failure rate = failed length inside maintenance "
            "map / all model pipe length matched to the maintenance map."
        ),
        (
            "  Jaccard = intersection length / "
            "(failed length + maintenance length - intersection length)."
        ),
        (
            "  Maintenance lines are coupled by exact maintenance pipe_name to "
            "model parent_link; model segment lengths are used for all percentages."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_root / "figureS4"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    city_data = []
    stats_rows = []
    for index, config in enumerate(CITY_CONFIGS, start=1):
        print(f"[{index}/3] Loading and classifying {config['name']} ...", flush=True)
        coords, failed, assessed, in_maintenance, stats = load_city(
            config,
            results_root,
            figure2_root=args.border_root.resolve(),
            years=args.years,
            wall_fraction=args.wall_fraction,
        )
        city_data.append((coords, failed, assessed, in_maintenance))
        stats_rows.append(stats)
        print(
            f"      {stats['failed_length_km']:,.2f} / "
            f"{stats['total_assessed_length_km']:,.2f} km failed "
            f"({stats['failure_percentage_of_assessed_length']:.2f}%)",
            flush=True,
        )

    print("Drawing Figure S4 ...", flush=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.0))
    for panel_label, ax, config, data, stats in zip(
        ("a", "b", "c"),
        axes,
        CITY_CONFIGS,
        city_data,
        stats_rows,
    ):
        plot_city(
            ax,
            config,
            *data,
            stats,
            border_root=args.border_root.resolve(),
            draw_borders=not args.no_borders,
            panel_label=panel_label,
        )

    legend_handles = [
        Line2D([0], [0], color=FAILED_COLOR, lw=2.0, label="Failed within 50 years"),
        Line2D([0], [0], color=NOT_FAILED_COLOR, lw=2.0, label="Not failed within 50 years"),
        Line2D([0], [0], color=MAINTENANCE_COLOR, lw=2.0, label="HK maintenance map"),
        Line2D(
            [0],
            [0],
            color=FAILED_MAINTENANCE_COLOR,
            lw=2.0,
            label="Failed + HK maintenance overlap",
        ),
    ]
    if any(int(row["unassessed_segments"]) for row in stats_rows):
        legend_handles.append(
            Line2D([0], [0], color=UNKNOWN_COLOR, lw=2.0, label="Unassessed")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=9,
    )
    fig.subplots_adjust(left=0.015, right=0.995, top=0.91, bottom=0.10, wspace=0.08)

    figure_path = output_dir / "FigureS4_50year_pipe_failure_600dpi.png"
    summary_path = output_dir / "FigureS4_50year_pipe_failure_length_summary.txt"
    maintenance_summary_path = (
        output_dir / "FigureS4_HK_maintenance_overlap_summary.txt"
    )
    fig.savefig(figure_path, dpi=DPI, format="png")
    write_summary(
        summary_path,
        stats_rows,
        years=args.years,
        wall_fraction=args.wall_fraction,
    )
    write_hk_maintenance_summary(
        maintenance_summary_path,
        stats_rows[0],
        years=args.years,
        wall_fraction=args.wall_fraction,
    )
    print(f"Figure written: {figure_path}", flush=True)
    print(f"Summary written: {summary_path}", flush=True)
    print(f"Maintenance overlap written: {maintenance_summary_path}", flush=True)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
