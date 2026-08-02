"""Shared node-level dry-weather-flow mixing utilities for the HRSNM models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.upper().isin(
        {"TRUE", "1", "YES", "T", "Y"}
    )


def build_node_dwf(
    nodes_df: pd.DataFrame,
    graph_nodes: Iterable[str],
) -> tuple[dict[str, float], list[str]]:
    """Validate ``nodes_all.csv`` data and return DWF by graph node.

    Missing DWF is accepted only for explicitly non-original split nodes.  A
    non-numeric, negative, duplicate, or graph-missing node is treated as an
    input-data error instead of being silently converted to zero.
    """
    required = {"node", "DWF_cms"}
    missing_columns = required - set(nodes_df.columns)
    if missing_columns:
        raise KeyError(f"nodes_all.csv missing required columns: {sorted(missing_columns)}")

    work = nodes_df.copy()
    work["node"] = work["node"].astype(str)
    duplicates = work.loc[work["node"].duplicated(keep=False), "node"].unique()
    if len(duplicates):
        sample = ", ".join(map(str, duplicates[:5]))
        raise ValueError(f"Duplicate node IDs in nodes_all.csv (sample: {sample})")

    raw = work["DWF_cms"]
    numeric = pd.to_numeric(raw, errors="coerce")
    nonnumeric = raw.notna() & numeric.isna()
    if nonnumeric.any():
        sample = ", ".join(work.loc[nonnumeric, "node"].head(5))
        raise ValueError(f"Non-numeric DWF_cms values (sample nodes: {sample})")

    if "is_original" in work.columns:
        is_original = _as_bool(work["is_original"])
    else:
        is_original = pd.Series(True, index=work.index)
    missing_original = numeric.isna() & is_original
    if missing_original.any():
        sample = ", ".join(work.loc[missing_original, "node"].head(5))
        raise ValueError(f"Original nodes with missing DWF_cms (sample: {sample})")

    numeric = numeric.fillna(0.0).astype(float)
    negative = numeric < 0.0
    if negative.any():
        sample = ", ".join(work.loc[negative, "node"].head(5))
        raise ValueError(f"Negative DWF_cms values (sample nodes: {sample})")
    work["DWF_cms"] = numeric

    graph_node_order = [str(node) for node in graph_nodes]
    graph_node_set = set(graph_node_order)
    table_node_set = set(work["node"])
    absent = [node for node in graph_node_order if node not in table_node_set]
    if absent:
        sample = ", ".join(absent[:5])
        raise ValueError(f"Graph nodes missing from nodes_all.csv (sample: {sample})")

    dwf_all = dict(zip(work["node"], work["DWF_cms"]))
    node_dwf = {node: float(dwf_all[node]) for node in graph_node_order}
    positive_in_table_order = [
        node
        for node, flow in zip(work["node"], work["DWF_cms"])
        if flow > 0.0 and node in graph_node_set
    ]
    return node_dwf, positive_in_table_order


def concentration_node_order(
    source_nodes: Sequence[str],
    positive_dwf_nodes: Sequence[str],
) -> list[str]:
    """Put legacy source nodes first, then append other local-DWF nodes."""
    ordered = [str(node) for node in source_nodes]
    seen = set(ordered)
    for node in positive_dwf_nodes:
        node = str(node)
        if node not in seen:
            ordered.append(node)
            seen.add(node)
    return ordered


def mix_node_concentration(
    incoming_flows: Sequence[float] | np.ndarray,
    incoming_concentrations: Sequence[Sequence[float]] | np.ndarray,
    local_flow: float,
    local_concentration: Sequence[float] | np.ndarray,
    *,
    n_components: int | None = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """Flow-weight incoming pipe outlets and local building wastewater."""
    flows = np.asarray(incoming_flows, dtype=float).reshape(-1)
    local = np.asarray(local_concentration, dtype=float).reshape(-1)
    if n_components is None:
        n_components = int(local.size)
    if local.size != n_components:
        raise ValueError("local_concentration has an unexpected component count")
    if not np.isfinite(local_flow) or local_flow < 0.0:
        raise ValueError("local_flow must be finite and non-negative")
    if np.any(~np.isfinite(flows)) or np.any(flows < 0.0):
        raise ValueError("incoming flows must be finite and non-negative")

    if flows.size:
        concentrations = np.asarray(incoming_concentrations, dtype=float)
        if concentrations.shape != (flows.size, n_components):
            raise ValueError(
                "incoming_concentrations must have shape "
                f"({flows.size}, {n_components})"
            )
        upstream_mass = (concentrations * flows[:, None]).sum(axis=0)
    else:
        upstream_mass = np.zeros(n_components, dtype=float)

    total_flow = float(flows.sum() + local_flow)
    if total_flow <= eps:
        return np.zeros(n_components, dtype=float)
    return (upstream_mass + local_flow * local) / total_flow


def apply_do_overrides(
    concentration: Sequence[float] | np.ndarray,
    *,
    air_dosage: bool = False,
    saturation_do: float | None = None,
    forced_do: float | None = None,
    do_index: int = 3,
) -> np.ndarray:
    """Apply air dosage first and an explicit node DO override last."""
    result = np.asarray(concentration, dtype=float).copy()
    if air_dosage:
        if saturation_do is None:
            raise ValueError("saturation_do is required when air_dosage is enabled")
        result[do_index] = float(saturation_do)
    if forced_do is not None:
        result[do_index] = float(forced_do)
    return result


def node_flow_balance_summary(
    pipes_df: pd.DataFrame,
    node_dwf: Mapping[str, float],
    *,
    start_col: str = "start",
    end_col: str = "end",
    flow_col: str = "flowrate",
    atol: float = 1e-12,
    rtol: float = 1e-8,
) -> dict[str, float | int]:
    """Audit hydraulic continuity at nodes that have outgoing segments."""
    starts = pipes_df[start_col].astype(str)
    ends = pipes_df[end_col].astype(str)
    flows = pd.to_numeric(pipes_df[flow_col], errors="coerce")
    if flows.isna().any() or (flows < 0.0).any():
        raise ValueError("Pipe flowrate contains missing, non-numeric, or negative values")

    incoming = flows.groupby(ends).sum()
    outgoing = flows.groupby(starts).sum()
    active_nodes = outgoing.index
    local = pd.Series(node_dwf, dtype=float).reindex(active_nodes, fill_value=0.0)
    expected = incoming.reindex(active_nodes, fill_value=0.0) + local
    residual = expected - outgoing
    denominator = np.maximum(outgoing.abs().to_numpy(), atol)
    relative = np.abs(residual.to_numpy()) / denominator
    within = np.isclose(
        expected.to_numpy(), outgoing.to_numpy(), atol=atol, rtol=rtol
    )

    indegree_nodes = set(ends)
    positive_nodes = [node for node, flow in node_dwf.items() if flow > 0.0]
    internal_positive = sum(node in indegree_nodes for node in positive_nodes)
    return {
        "n_graph_nodes": int(len(node_dwf)),
        "n_positive_dwf_nodes": int(len(positive_nodes)),
        "n_internal_positive_dwf_nodes": int(internal_positive),
        "total_dwf_cms": float(sum(node_dwf.values())),
        "n_active_nodes": int(len(active_nodes)),
        "n_flow_balance_failures": int((~within).sum()),
        "max_abs_flow_balance_residual_cms": float(np.abs(residual).max()),
        "max_rel_flow_balance_residual": float(relative.max()),
    }
