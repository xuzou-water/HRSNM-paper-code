"""Shared dissolved-oxygen saturation calculations for HRSNM."""

from __future__ import annotations

import math


MIN_TEMPERATURE_C = 0.0
MAX_TEMPERATURE_C = 40.0


def saturation_do_mg_l(temp_c: float) -> float:
    """Return freshwater oxygen saturation at 1 atm in mg/L.

    This is equation 7 of USGS Technical Memorandum 2011.03, based on
    Benson and Krause (1980, 1984). The published equation is valid from
    0 to 40 degrees Celsius.
    """
    try:
        temperature_c = float(temp_c)
    except (TypeError, ValueError) as exc:
        raise ValueError("temp_c must be a finite number") from exc

    if not math.isfinite(temperature_c):
        raise ValueError("temp_c must be a finite number")
    if not MIN_TEMPERATURE_C <= temperature_c <= MAX_TEMPERATURE_C:
        raise ValueError(
            f"temp_c must be between {MIN_TEMPERATURE_C:g} and "
            f"{MAX_TEMPERATURE_C:g} degrees Celsius"
        )

    temperature_k = temperature_c + 273.15
    return math.exp(
        -139.34411
        + 1.575701e5 / temperature_k
        - 6.642308e7 / temperature_k**2
        + 1.243800e10 / temperature_k**3
        - 8.621949e11 / temperature_k**4
    )
