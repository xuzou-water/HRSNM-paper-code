"""Corrosion criteria shared by the HRSNM figure scripts."""

from __future__ import annotations

import numpy as np


DEFAULT_CORROSION_RATE_THRESHOLD_MM_PER_YEAR = 1.0
DEFAULT_CORROSION_HORIZON_YEARS = 50.0
DEFAULT_WALL_THICKNESS_FRACTION = 0.075


def corrosion_exceeds_assumed_wall(
    corrosion_rate_mm_per_year,
    diameter_m,
    *,
    years: float = DEFAULT_CORROSION_HORIZON_YEARS,
    wall_fraction: float = DEFAULT_WALL_THICKNESS_FRACTION,
):
    """Return whether cumulative corrosion exceeds assumed pipe-wall thickness.

    Invalid/non-positive diameters and non-finite or negative corrosion rates
    return ``False``. Equality is not classified as failure.
    """
    if not np.isfinite(years) or years <= 0:
        raise ValueError("years must be a positive finite number")
    if not np.isfinite(wall_fraction) or not 0 < wall_fraction < 1:
        raise ValueError("wall_fraction must be between 0 and 1")

    rate = np.asarray(corrosion_rate_mm_per_year, dtype=float)
    diameter = np.asarray(diameter_m, dtype=float)
    cumulative_corrosion_mm = rate * years
    assumed_wall_thickness_mm = diameter * 1000.0 * wall_fraction
    valid = (
        np.isfinite(rate)
        & np.isfinite(diameter)
        & (rate >= 0)
        & (diameter > 0)
    )
    result = valid & (cumulative_corrosion_mm > assumed_wall_thickness_mm)
    if result.ndim == 0:
        return bool(result)
    return result


def corrosion_rate_exceeds_threshold(
    corrosion_rate_mm_per_year,
    threshold_mm_per_year: float = DEFAULT_CORROSION_RATE_THRESHOLD_MM_PER_YEAR,
):
    """Return ``True`` where a finite corrosion rate strictly exceeds threshold.

    The comparison is intentionally direct: no design horizon, pipe diameter,
    wall-thickness fraction, or hydraulic water depth is involved.
    """
    threshold = float(threshold_mm_per_year)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold_mm_per_year must be finite and non-negative")

    rate = np.asarray(corrosion_rate_mm_per_year, dtype=float)
    result = np.isfinite(rate) & (rate > threshold)
    if result.ndim == 0:
        return bool(result)
    return result


def corrosion_exceedance_length_km(
    corrosion_rate_mm_per_year,
    pipe_length_m,
    threshold_mm_per_year: float = DEFAULT_CORROSION_RATE_THRESHOLD_MM_PER_YEAR,
) -> float:
    """Sum valid positive pipe lengths whose corrosion rate exceeds threshold."""
    rate = np.asarray(corrosion_rate_mm_per_year, dtype=float)
    length = np.asarray(pipe_length_m, dtype=float)
    if rate.shape != length.shape:
        raise ValueError("corrosion-rate and pipe-length arrays must have equal shape")
    exceedance = corrosion_rate_exceeds_threshold(rate, threshold_mm_per_year)
    valid_length = np.isfinite(length) & (length > 0)
    return float(length[exceedance & valid_length].sum() / 1000.0)
