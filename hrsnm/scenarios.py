"""Shared environmental scenario grids for local and server HRSNM runs."""

from __future__ import annotations


SCENARIO_GRIDS = {
    # Eight corner combinations for low-memory local workflow validation.
    "small8": {
        "temperatures": (15, 25),
        "so4": (5, 25),
        "cod": (250, 800),
    },
    # Complete manuscript grid for the Ubuntu server run.
    "full27": {
        "temperatures": (15, 20, 25),
        "so4": (5, 15, 25),
        "cod": (250, 525, 800),
    },
}


def scenario_axes(name: str):
    """Return temperature, sulphate and COD axes for a named grid."""

    try:
        grid = SCENARIO_GRIDS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown scenario grid {name!r}; expected one of {tuple(SCENARIO_GRIDS)}"
        ) from exc
    return grid["temperatures"], grid["so4"], grid["cod"]


def scenario_count(name: str) -> int:
    axes = scenario_axes(name)
    return len(axes[0]) * len(axes[1]) * len(axes[2])
