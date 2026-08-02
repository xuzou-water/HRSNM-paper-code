# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:34:04 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:59:22 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:24:43 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
LA Sewer Network Water Quality Simulation & H2S Emission Analysis
=================================================================
读取方式与代码1(HK)完全一致：直接读取 segment_hydraulics.csv + nodes_all.csv，
不切分、不聚合，所有结果以 segment 为单位输出。

Baseline:
    Temperature = 26 °C
    COD mean    = 399.17, COD_CI = 39.917 (10% of mean)
    SO4 mean    = 20,     SO4_CI = 10

NOTE (本次修改):
    计算 H2S 溢出 distance 时，只对满足
        us_is_original OR ds_is_original OR (起点) is_original
    的 segment 进行计算，其余 distance 置为 NaN。
"""

import argparse
import os
import time
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from hrsnm_dissolved_oxygen import saturation_do_mg_l
from hrsnm_node_mixing import (
    apply_do_overrides,
    build_node_dwf,
    concentration_node_order,
    mix_node_concentration,
    node_flow_balance_summary,
)
from hrsnm_scenarios import scenario_axes

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# 开关
# ═══════════════════════════════════════════════════════════════════════════════
RUN_SCENARIO_ANALYSIS = False


# ═══════════════════════════════════════════════════════════════════════════════
# 路径 (LA)
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "processed_data", "la_v3")
HYD_DIR = os.path.join(DATA_DIR, "hydraulic_results")
DEFAULT_OUT_DIR = os.path.join(
    SCRIPT_DIR, "node_dwf_fix_results", "LA_v3", "biochemical_results"
)
OUT_DIR = DEFAULT_OUT_DIR

# 只需要 segment 水力 + 节点
SEG_HYD_CSV = os.environ.get(
    "HRSNM_LA_SEGMENTS_CSV", os.path.join(HYD_DIR, "segment_hydraulics.csv")
)
NODES_ALL_CSV = os.environ.get(
    "HRSNM_LA_NODES_CSV", os.path.join(HYD_DIR, "nodes_all.csv")
)


# ═══════════════════════════════════════════════════════════════════════════════
# 全局常量 (生化 / 气相 / 腐蚀)
# ═══════════════════════════════════════════════════════════════════════════════
ALPHA, BETA = 1, 1
K_1_2 = 6
KSF = 5
AF = 1.05

UHO2, KSW, KO = 4, 1, 0.5
AW = 1.07

# --- LA 基线温度 ---
TEMP = 26.0

YHF, YHW = 0.55, 0.55
QM, KH1, KX1 = 1, 5, 1.5
E_PARAM, XHF = 0.15, 10
KH2, KX2, DHANA = 0.5, 0.5, 0.1
A_PARAM = 1.05

KCH4H2, KH2MA, KACMA = 1.92, 0.002, 1
QACETOG, KF, QACIDOG = 12.2, 10, 2.14
KH2SH2, KH2SRB, KACSRB, KPROP = 0.001, 0.01, 5, 5

K_I_SO4, KSO4 = 4.9258135, 62.851516
KH2SAC = 0.43105076 * 24
KH2SPROP = 0.31158032 * 24
KCH4AC = 0.44108655 * 24

KS2C, KS2B = 0.42, 0.99
N1, N2 = 1, 0.1
K_S_OX_F = 0.085
R_CWC, R_CWB, R_CFB = 0.85, 2, 0.5

F_XHW = 0.11
F_SF  = 0.22
F_SA  = 0.26
F_XS1 = 0.14
F_XS2 = 1-F_XHW-F_SF-F_SA-F_XS1

# --- LA 基线浓度参数 ---
SO4_MEAN_LA = 20.0
SO4_CI_LA   = 10.0

COD_PER_SOURCE_LA = 399.17
COD_CI_LA         = 399.17 * 0.10   # = 39.917

SO4_CV_BASELINE = SO4_CI_LA / SO4_MEAN_LA
COD_CV_BASELINE = COD_CI_LA / COD_PER_SOURCE_LA

CONC_KEYS = [
    'XHw', 'Xs1', 'Xs2', 'SO', 'SF', 'Sac',
    'SHS', 'SSO4', 'CH4', 'Sprop', 'H2'
]
N_CONC = len(CONC_KEYS)

G_GRAVITY = 9.81
R_GAS = 8.314
T_GAS = 293.15

H_A_H2S = 560   * np.exp(-2200 * (1 / T_GAS - 1 / 298))
H_A_CH4 = 40200 * np.exp(-2200 * (1 / T_GAS - 1 / 298))

N_CORR = 0.54
KN_CORR = 9.6
# Pomeroy acid-formation factor for moderate climates (Jiang et al., 2014;
# https://doi.org/10.1016/j.watres.2014.07.026).
CORR_VALUE = 0.8
CONC_MASS = 2_400_000
A_CO = 0.2

RHO_AIR = 1.2
M_H2S = 34.0
M_AIR = 29.0
U_WIND = 1.0
SIGMA_Y_A = 0.412
SIGMA_Z_C = 0.311
SIGMA_B = 0.7

D_PICK = 0.025
A_PICK = np.pi * (D_PICK / 2.0) ** 2
C_D_ORIFICE = 0.62
RHO_GAS = 1.2

EPS = 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# 工具：布尔列规范化
# ═══════════════════════════════════════════════════════════════════════════════
def _to_bool_series(series):
    """把各种写法 (TRUE/True/true/1/yes/T) 统一转换为 bool, NaN -> False。"""
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
              .str.strip()
              .str.upper()
              .isin(['TRUE', '1', 'YES', 'T'])
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — 读取 segment / node  (直读, 不再拼接 mid 节点)
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_read_csv(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {name}: {path}")
    return pd.read_csv(path)


def load_segments(seg_csv=SEG_HYD_CSV, nodes_csv=NODES_ALL_CSV):
    """
    直接读取:
      - segment_hydraulics.csv  每根 segment 的水力 + 几何 + 端点节点
      - nodes_all.csv           所有节点坐标 / invert / type / is_original
    """
    print("  Reading segment_hydraulics.csv ...")
    seg = _safe_read_csv(seg_csv, "segment_hydraulics.csv")
    print(f"    segments: {len(seg):,}")

    print("  Reading nodes_all.csv ...")
    nodes_df = _safe_read_csv(nodes_csv, "nodes_all.csv")
    nodes_df['node'] = nodes_df['node'].astype(str)
    if 'is_original' in nodes_df.columns:
        nodes_df['is_original'] = _to_bool_series(nodes_df['is_original'])
    else:
        print("    ⚠ nodes_all.csv 缺少 is_original 列，默认全部 False。")
        nodes_df['is_original'] = False
    print(f"    nodes  : {len(nodes_df):,}")

    # 类型规范
    for c in ['link_name', 'parent_link', 'us_node', 'ds_node',
              'pipe_type', 'catchment', 'regime']:
        if c in seg.columns:
            seg[c] = seg[c].astype(str)

    # us/ds 是否原始节点 (来自 segment_hydraulics.csv)
    if 'us_is_original' in seg.columns:
        seg['us_is_original'] = _to_bool_series(seg['us_is_original'])
    else:
        print("    ⚠ segment_hydraulics.csv 缺少 us_is_original 列，默认全部 False。")
        seg['us_is_original'] = False

    if 'ds_is_original' in seg.columns:
        seg['ds_is_original'] = _to_bool_series(seg['ds_is_original'])
    else:
        print("    ⚠ segment_hydraulics.csv 缺少 ds_is_original 列，默认全部 False。")
        seg['ds_is_original'] = False

    # 重命名 → name/start/end/v/depth/flowrate
    seg = seg.rename(columns={
        'link_name':  'name',
        'us_node':    'start',
        'ds_node':    'end',
        'length_m':   'length',
        'diameter_m': 'diameter',
        'V_mps':      'v',
        'h_m':        'depth',
        'Q_cms':      'flowrate',
        'us_invert':  'In_elev',
        'ds_invert':  'Out_elev',
    })

    for c in ['length', 'diameter', 'v', 'depth', 'flowrate', 'slope',
              'us_x', 'us_y', 'ds_x', 'ds_y', 'In_elev', 'Out_elev']:
        if c in seg.columns:
            seg[c] = pd.to_numeric(seg[c], errors='coerce').fillna(0.0)

    seg['depth']    = np.clip(seg['depth'], 0.0, seg['diameter'])
    seg['v']        = seg['v'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    seg['flowrate'] = seg['flowrate'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"    segments (模拟单元): {len(seg):,}")
    return seg, nodes_df


def attach_original_flags(pipes, nodes_df):
    """
    把起点节点 (start = us_node) 的 is_original 合并进 segment，
    生成列 node_is_original。同时确保 us_is_original / ds_is_original 为 bool。
    """
    pipes = pipes.copy()

    if 'us_is_original' not in pipes.columns:
        pipes['us_is_original'] = False
    if 'ds_is_original' not in pipes.columns:
        pipes['ds_is_original'] = False
    pipes['us_is_original'] = _to_bool_series(pipes['us_is_original'])
    pipes['ds_is_original'] = _to_bool_series(pipes['ds_is_original'])

    node_orig = (
        nodes_df[['node', 'is_original']]
        .rename(columns={'node': 'start', 'is_original': 'node_is_original'})
        .drop_duplicates(subset='start')
    )
    pipes = pipes.merge(node_orig, on='start', how='left')
    pipes['node_is_original'] = _to_bool_series(
        pipes['node_is_original'].fillna(False)
    )

    n_orig = (pipes['us_is_original'] |
              pipes['ds_is_original'] |
              pipes['node_is_original']).sum()
    print(f"  原始节点相关 segment 数 (将参与 distance 计算): {n_orig:,} / {len(pipes):,}")
    return pipes


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — 几何
# ═══════════════════════════════════════════════════════════════════════════════
def compute_geometry(pipes):
    pipes = pipes.copy()
    D = pd.to_numeric(pipes['diameter'], errors='coerce').fillna(0.0).values.astype(float)
    h = pd.to_numeric(pipes['depth'],    errors='coerce').fillna(0.0).values.astype(float)
    L = pd.to_numeric(pipes['length'],   errors='coerce').fillna(0.0).values.astype(float)
    v = pd.to_numeric(pipes['v'],        errors='coerce').fillna(0.0).values.astype(float)

    h = np.clip(h, 0, D)
    R = D / 2

    theta = np.zeros_like(D)
    valid = R > 0
    theta[valid] = 2 * np.arccos(np.clip((R[valid] - h[valid]) / R[valid], -1, 1))

    A_w = (R**2 / 2) * (theta - np.sin(theta))
    P_w = R * theta
    total_peri = 2 * np.pi * R
    total_area = np.pi * R**2
    Ag = (total_peri - P_w) * L
    Vg = (total_area - A_w) * L
    Vw = A_w * L

    pipes['A_V']   = np.where((P_w == 0) | (A_w == 0), 0, np.minimum(P_w / A_w, 100))
    pipes['A_V_g'] = np.where((Ag == 0) | (Vg == 0), 0, Ag / np.maximum(Vg, EPS))
    pipes['HRT']   = np.where(v == 0, 0, (L / np.maximum(v, EPS)) / 3600)
    pipes['A'], pipes['P'] = A_w, P_w
    pipes['Ag'], pipes['Vg'], pipes['Vw'] = Ag, Vg, Vw
    pipes['depth'] = h
    return pipes


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3 — 拓扑 (segment 级)
# ═══════════════════════════════════════════════════════════════════════════════
def build_topology(pipes_df, removed_edges_csv=None):
    G = nx.DiGraph()
    edge_lookup = {}

    for _, row in pipes_df.iterrows():
        s, e = str(row['start']), str(row['end'])
        name = str(row['name'])
        if s == e:
            continue
        G.add_edge(s, e, name=name)
        edge_lookup[(s, e)] = row.to_dict()

    removed_edges = []

    if not nx.is_directed_acyclic_graph(G):
        print("  ⚠ 图中存在环路,正在移除回边 ...")
        removed = 0
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G)
                u, v = cycle[-1][0], cycle[-1][1]
                edge_info = edge_lookup.get((u, v), {})
                removed_edges.append({
                    'start': u, 'end': v,
                    'name': edge_info.get('name', ''),
                    'length': edge_info.get('length', np.nan),
                    'diameter': edge_info.get('diameter', np.nan),
                    'reason': 'Removed to break directed cycle'
                })
                G.remove_edge(u, v)
                removed += 1
            except nx.NetworkXNoCycle:
                break
        print(f"    removed {removed} back-edges")

        if removed_edges_csv is not None and len(removed_edges) > 0:
            pd.DataFrame(removed_edges).to_csv(
                removed_edges_csv, index=False, encoding='utf-8-sig'
            )

    source_nodes = [n for n in G.nodes if G.in_degree(n) == 0]

    node_levels = {}
    for nd in nx.topological_sort(G):
        preds = list(G.predecessors(nd))
        node_levels[nd] = 0 if not preds else max(node_levels[p] for p in preds) + 1

    level_to_nodes = defaultdict(list)
    for nd, lv in node_levels.items():
        level_to_nodes[lv].append(nd)
    max_level = max(node_levels.values()) if node_levels else 0

    pipe_name2idx = {str(n): i for i, n in enumerate(pipes_df['name'].astype(str))}
    node2idx = {str(n): i for i, n in enumerate(G.nodes())}
    idx2node = {i: n for n, i in node2idx.items()}

    n_pipes = len(pipes_df)
    start_idx = np.zeros(n_pipes, dtype=np.int32)
    end_idx   = np.zeros(n_pipes, dtype=np.int32)
    node_out, node_in = defaultdict(list), defaultdict(list)

    valid_pipe_mask = np.zeros(n_pipes, dtype=bool)
    for _, row in pipes_df.iterrows():
        pid = pipe_name2idx[str(row['name'])]
        s, e = str(row['start']), str(row['end'])
        if G.has_edge(s, e):
            start_idx[pid], end_idx[pid] = node2idx[s], node2idx[e]
            node_out[s].append(pid)
            node_in[e].append(pid)
            valid_pipe_mask[pid] = True

    print(f"  Graph: {n_pipes:,} segments, {len(G.nodes()):,} nodes, "
          f"{len(source_nodes):,} sources, {max_level + 1:,} levels")
    print(f"  Active segments    : {valid_pipe_mask.sum():,}")

    return (G, source_nodes, level_to_nodes, max_level,
            pipe_name2idx, node2idx, idx2node,
            start_idx, end_idx, node_out, node_in, node_levels,
            valid_pipe_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 4 — 生化反应 (RK4)
# ═══════════════════════════════════════════════════════════════════════════════
def _derivatives(conc, A_V, vel, diam, slope, temp_c, is_force_main=None):
    aw_temp = AW ** (temp_c - 20)
    a_temp  = A_PARAM ** (temp_c - 20)
    af_temp = AF ** (temp_c - 20)
    saturation_do = saturation_do_mg_l(temp_c)

    XHw = np.maximum(conc[:, 0], EPS)
    Xs1, Xs2 = conc[:, 1], conc[:, 2]
    SO, SF, Sac = conc[:, 3], conc[:, 4], conc[:, 5]
    SHS, SSO4 = conc[:, 6], conc[:, 7]
    CH4, Sprop, H2 = conc[:, 8], conc[:, 9], conc[:, 10]

    dm = np.maximum(np.pi / 8 * diam, EPS)
    Fr = vel * (G_GRAVITY * dm) ** (-0.5)
    su = np.maximum(np.maximum(slope, 0) * vel, EPS)
    K_L = 0.86 * (1 + 0.2 * Fr**2) * su**(3 / 8) * dm**(-1)

    rgrw   = UHO2 * (SF + Sac) / (KSW + SF + Sac + EPS) * SO / (KO + SO + EPS) * XHw * aw_temp
    rgrf   = K_1_2 * SO**0.5 * YHF / (1 - YHF) * (SF + Sac) / (KSF + SF + Sac + EPS) * A_V * af_temp
    rmaint = QM * SO / (KO + SO + EPS) * XHw * aw_temp
    rhydr1 = KH1 * (Xs1 / XHw) / (KX1 + Xs1 / XHw + EPS) * SO / (KO + SO + EPS) \
             * (XHw + E_PARAM * XHF * A_V) * aw_temp
    rhydr2 = KH2 * (Xs2 / XHw) / (KX2 + Xs2 / XHw + EPS) * SO / (KO + SO + EPS) \
             * (XHw + E_PARAM * XHF * A_V) * aw_temp
    rd     = DHANA * KO / (KO + SO + EPS) * XHw * a_temp
    ra     = ALPHA * K_L * (BETA * saturation_do - SO) * a_temp * 24

    # ★ rising main (force main / 压力管) 不考虑复氧 ra = 0
    if is_force_main is not None:
        ra = np.where(is_force_main, 0.0, ra)

    inh     = K_I_SO4 / (SSO4 + K_I_SO4 + SHS + EPS)
    anaerob = KO / (KO + SO + EPS)

    R1 = KCH4H2  * H2  / (KH2MA  + H2  + EPS) * A_V * a_temp * anaerob * inh
    R2 = KCH4AC  * Sac / (KACMA  + Sac + EPS) * A_V * a_temp * anaerob * inh
    R3 = QACETOG * SF  / (KF     + SF  + EPS) * A_V * a_temp * anaerob
    R4 = QACIDOG * SF  / (KF     + SF  + EPS) * A_V * a_temp * anaerob
    R5 = KH2SH2  * H2  / (KH2SRB + H2  + EPS) * SSO4 / (KSO4 + SSO4 + EPS) * A_V * a_temp * anaerob
    R6 = KH2SAC  * Sac / (KACSRB + Sac + EPS) * SSO4 / (KSO4 + SSO4 + EPS) * A_V * a_temp * anaerob
    R7 = KH2SPROP* Sprop/(KPROP  + Sprop+ EPS) * SSO4 / (KSO4 + SSO4 + EPS) * A_V * a_temp * anaerob

    SHS_p, SO_p = np.maximum(SHS, 0), np.maximum(SO, 0)
    rs2c  = KS2C * SHS_p**N1 * SO_p**N2 * 24
    rs2b  = KS2B * SHS_p**N1 * SO_p**N2 * 24
    rs2_f = K_S_OX_F * SHS_p**0.5 * SO_p**0.5 * 24 * A_V

    d = np.zeros_like(conc)
    d[:, 0] = rgrw + rgrf - rmaint - rd
    d[:, 1] = -rhydr1
    d[:, 2] = -rhydr2 + rd
    d[:, 3] = (ra - (1 - YHW) / YHW * rgrw - (1 - YHF) / YHF * rgrf - rmaint
               - rs2c / R_CWC - rs2b / R_CWB - rs2_f / R_CFB)
    d[:, 4] = (-R3 - R4 - rgrw / (2 * YHW) - rgrf / (2 * YHF) - 0.5 * rmaint
               + rhydr1 + rhydr2)
    d[:, 5] = (-R2 + 2 / 3 * R3 + 0.2222 * R4 - 2 * R6 + 2.6667 * R7
               - rgrw / (2 * YHW) - rgrf / (2 * YHF) - 0.5 * rmaint)
    d[:, 6] = R5 + R6 + R7 - rs2c - rs2b - rs2_f
    d[:, 7] = -R5 - R6 - R7 + rs2c + rs2b + rs2_f
    d[:, 8] = R1 + R2
    d[:, 9] = 0.7778 * R4 - 4.6666 * R7
    d[:, 10] = -R1 + R3 / 3 - 2 * R5
    return d


def _rk4_step(conc, A_V, vel, diam, slope, dt, temp_c, is_force_main=None):
    dt2 = dt[:, np.newaxis]
    k1 = _derivatives(conc, A_V, vel, diam, slope, temp_c, is_force_main)
    k2 = _derivatives(np.maximum(conc + 0.5 * dt2 * k1, 0), A_V, vel, diam, slope, temp_c, is_force_main)
    k3 = _derivatives(np.maximum(conc + 0.5 * dt2 * k2, 0), A_V, vel, diam, slope, temp_c, is_force_main)
    k4 = _derivatives(np.maximum(conc + dt2 * k3, 0), A_V, vel, diam, slope, temp_c, is_force_main)
    return np.maximum(conc + (dt2 / 6) * (k1 + 2 * k2 + 2 * k3 + k4), 0)


def compute_reactions_batch(conc, A_V, HRT, vel, diam, slope, temp_c, is_force_main=None):
    out = conc.copy()
    mask = (HRT > 0) & (A_V > 0)
    if not mask.any():
        return out
    idx = np.where(mask)[0]
    ifm = None if is_force_main is None else is_force_main[idx]
    out[idx] = _rk4_step(
        conc[idx], A_V[idx], vel[idx], diam[idx], slope[idx],
        HRT[idx] / 24.0, temp_c, ifm
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Part 5 — 源浓度 & 主循环
# ═══════════════════════════════════════════════════════════════════════════════
def generate_source_concentrations_la(source_nodes,
                                      so4_mean=SO4_MEAN_LA,
                                      cod_mean=COD_PER_SOURCE_LA,
                                      so4_ci=SO4_CI_LA,
                                      cod_ci=COD_CI_LA):
    np.random.seed(20)
    sc = np.zeros((len(source_nodes), N_CONC))
    for i, _ in enumerate(source_nodes):
        s_so4 = max(np.random.normal(so4_mean, so4_ci), 0)
        s_cod = max(np.random.normal(cod_mean, cod_ci), 0)
        s_do  = np.random.uniform(0, 4)
        sc[i] = [
            s_cod * F_XHW, s_cod * F_XS1, s_cod * F_XS2, s_do,
            s_cod * F_SF, s_cod * F_SA, 0, s_so4, 0, 0, 0
        ]
    return sc


def run_simulation_la(pipes_df, topo, node_dwf, positive_dwf_nodes,
                      so4_mean=SO4_MEAN_LA,
                      cod_mean=COD_PER_SOURCE_LA,
                      so4_ci=SO4_CI_LA,
                      cod_ci=COD_CI_LA,
                      air_dosage_nodes=None,
                      temp_c=TEMP):
    saturation_do = saturation_do_mg_l(temp_c)
    (G, source_nodes, level_to_nodes, max_level,
     pipe_name2idx, node2idx, idx2node,
     start_idx, end_idx, node_out, node_in, _node_levels,
     valid_pipe_mask) = topo

    n_pipes, n_nodes = len(pipes_df), len(G.nodes())
    conc_in   = np.zeros((n_pipes, N_CONC))
    conc_out  = np.zeros((n_pipes, N_CONC))
    node_conc = np.zeros((n_nodes, N_CONC))
    processed = np.zeros(n_pipes, dtype=bool)

    AV  = pipes_df['A_V'].values.astype(np.float64)
    HRT = pipes_df['HRT'].values.astype(np.float64)
    v   = pipes_df['v'].values.astype(np.float64)
    fl  = pipes_df['flowrate'].values.astype(np.float64)
    sl  = pipes_df['slope'].values.astype(np.float64)
    dm  = pipes_df['diameter'].values.astype(np.float64)

    # ★ 压力管(force_main)掩码：用 pipe_type 区分
    if 'pipe_type' in pipes_df.columns:
        is_fm = pipes_df['pipe_type'].astype(str).str.strip().str.lower().eq('force_main').values
    else:
        is_fm = np.zeros(len(pipes_df), dtype=bool)

    concentration_nodes = concentration_node_order(source_nodes, positive_dwf_nodes)
    local_inflow_conc = generate_source_concentrations_la(
        concentration_nodes,
        so4_mean=so4_mean, cod_mean=cod_mean,
        so4_ci=so4_ci, cod_ci=cod_ci
    )
    concentration_index = {nd: i for i, nd in enumerate(concentration_nodes)}
    zero_conc = np.zeros(N_CONC, dtype=float)

    t0 = time.time()

    for level in tqdm(range(max_level + 1), desc="Processing levels", leave=False):
        nodes = level_to_nodes[level]
        if not nodes:
            continue

        for nd in nodes:
            inc = np.asarray(node_in.get(nd, []), dtype=int)
            inc = inc[valid_pipe_mask[inc]] if len(inc) else inc
            if len(inc) and not processed[inc].all():
                raise RuntimeError(
                    f"Node {nd} reached before all incoming pipes were processed"
                )
            local_flow = float(node_dwf.get(nd, 0.0))
            local_conc = (
                local_inflow_conc[concentration_index[nd]]
                if local_flow > 0.0 else zero_conc
            )
            mixed = mix_node_concentration(
                fl[inc], conc_out[inc], local_flow, local_conc,
                n_components=N_CONC, eps=EPS,
            )
            node_conc[node2idx[nd]] = apply_do_overrides(
                mixed,
                air_dosage=bool(air_dosage_nodes and nd in air_dosage_nodes),
                saturation_do=saturation_do,
            )

        pids, pmap = [], {}
        for nd in nodes:
            for pid in node_out.get(nd, []):
                if valid_pipe_mask[pid]:
                    pids.append(pid)
                    pmap[pid] = nd

        if not pids:
            continue

        pids = np.array(pids, dtype=int)
        inlet = np.array([node_conc[node2idx[pmap[p]]] for p in pids])
        conc_in[pids] = inlet

        outlet = compute_reactions_batch(
            inlet, AV[pids], HRT[pids], v[pids], dm[pids], sl[pids], temp_c,
            is_fm[pids]          # ★ 传入 force_main 掩码
        )
        conc_out[pids] = outlet
        processed[pids] = True

    print(f"  Simulation time: {time.time() - t0:.2f}s")
    return conc_in, conc_out, node_conc


# ═══════════════════════════════════════════════════════════════════════════════
# Part 6 — 后处理 & 保存
# ═══════════════════════════════════════════════════════════════════════════════


def postprocess_and_save_la(pipes_split, conc_in, conc_out, tag="", temp_c=TEMP):
    p = pipes_split.copy()
    suffix = f"_{tag}" if tag else ""

    for i, k in enumerate(CONC_KEYS):
        p[f'{k}_in']  = conc_in[:, i]
        p[f'{k}_out'] = conc_out[:, i]
    p['temp_c'] = temp_c

    saturation_do = saturation_do_mg_l(temp_c)
    a_temp   = A_PARAM ** (temp_c - 20)
    anaerob  = KO / (KO + p['SO_in'] + EPS)
    so4_term = p['SSO4_in'] / (KSO4 + p['SSO4_in'] + EPS)
    av       = p['A_V']

    p['R5'] = KH2SH2 * p['H2_in'] / (KH2SRB + p['H2_in'] + EPS) * so4_term * av * a_temp * anaerob
    p['R6'] = KH2SAC * p['Sac_in'] / (KACSRB + p['Sac_in'] + EPS) * so4_term * av * a_temp * anaerob
    p['R7'] = KH2SPROP * p['Sprop_in'] / (KPROP + p['Sprop_in'] + EPS) * so4_term * av * a_temp * anaerob

    shs_p = np.maximum(p['SHS_in'], 0)
    so_p  = np.maximum(p['SO_in'], 0)
    p['rs2c']     = KS2C * shs_p**N1 * so_p**N2 * 24
    p['rs2b']     = KS2B * shs_p**N1 * so_p**N2 * 24
    p['rs2_ox_f'] = K_S_OX_F * shs_p**0.5 * so_p**0.5 * 24 * av

    # ★ 新增：计算复氧速率 ra（与 _derivatives 中公式完全一致，使用 *_in 浓度）
    vel   = p['v'].values.astype(float)
    diam  = p['diameter'].values.astype(float)
    slope = p['slope'].values.astype(float)
    SO_in = p['SO_in'].values.astype(float)

    dm  = np.maximum(np.pi / 8 * diam, EPS)
    Fr  = vel * (G_GRAVITY * dm) ** (-0.5)
    su  = np.maximum(np.maximum(slope, 0) * vel, EPS)
    K_L = 0.86 * (1 + 0.2 * Fr**2) * su**(3 / 8) * dm**(-1)
    p['ra'] = ALPHA * K_L * (BETA * saturation_do - SO_in) * a_temp * 24
    # ★ force_main(压力管) 复氧 ra = 0
    if 'pipe_type' in p.columns:
        is_fm = p['pipe_type'].astype(str).str.strip().str.lower().eq('force_main').values
        p.loc[is_fm, 'ra'] = 0.0
    # ★ 新增结束

    out_path = os.path.join(OUT_DIR, f"la_result_segments_v7{suffix}.csv")
    p.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Segment-level WQ results → {out_path}")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Part 7 — H2S / CH4 气相、腐蚀、大气扩散
# ═══════════════════════════════════════════════════════════════════════════════
def calc_headspace_h2s_ppm_from_total_sulfide(
    shs_g_m3, Vw_m3, Vg_m3, pH=7.5, T_K=298.15,
    pKa1=7.04, H_cp=0.102, MW_S=32.065
):
    shs_g_m3 = np.asarray(shs_g_m3, dtype=float)
    Vw_m3 = np.asarray(Vw_m3, dtype=float)
    Vg_m3 = np.asarray(Vg_m3, dtype=float)

    R_atm = 0.082057
    V_L = np.maximum(Vw_m3, 0.0) * 1000.0
    V_G = np.maximum(Vg_m3, 0.0) * 1000.0

    C_S_tot = np.maximum(shs_g_m3, 0.0) / 1000.0 / MW_S
    n_tot = C_S_tot * V_L
    alpha = 10.0 ** (pH - pKa1)

    denom = H_cp * V_L * (1.0 + alpha) + V_G / (R_atm * T_K)
    P_H2S = np.where(denom > 0.0, n_tot / denom, 0.0)

    ppmv = P_H2S * 1e6
    C_H2S_aq = H_cp * P_H2S
    sh2s_w_g_m3_as_S = C_H2S_aq * MW_S * 1000.0

    return ppmv, sh2s_w_g_m3_as_S


def compute_h2s_emission_la(pipes_split, tag="", temp_c=TEMP):
    suffix = f"_{tag}" if tag else ""
    fp = pipes_split.copy()

    # 缺失的原始节点标志列做兜底
    for col in ['us_is_original', 'ds_is_original', 'node_is_original']:
        if col not in fp.columns:
            fp[col] = False
        fp[col] = _to_bool_series(fp[col])

    df = pd.DataFrame({
        'pipe_name': fp['name'].values,
        'node_name': fp['start'].values,
        'SHS': fp['SHS_in'].values, #[g S/m³]
        'velocity': fp['v'].values,
        'depth': fp['depth'].values,
        'diameter': fp['diameter'].values,
        'slope': fp['slope'].values,
        'Vg': fp['Vg'].values,
        'Vw': fp['Vw'].values,
        'Ag': fp['Ag'].values,
        'CH4': fp['CH4_in'].values,
        'us_x': fp['us_x'].values,
        'us_y': fp['us_y'].values,
        'HRT': fp['HRT'].values,
        'SO': fp['SO_in'].values,
        'XHw': fp['XHw_in'].values,
        'SF': fp['SF_in'].values,
        'Sac': fp['Sac_in'].values,
        'length': fp['length'].values,
        'flowrate': fp['flowrate'].values * 86400,
        'temp_c': temp_c,
        'us_is_original': fp['us_is_original'].values,
        'ds_is_original': fp['ds_is_original'].values,
        'node_is_original': fp['node_is_original'].values,
    })

    PH_H2S = 7.5
    T_H2S_K = temp_c + 273.15
    PKA1_H2S = 7.04
    HCP_H2S = 0.102

    df['SH2S'], df['SH2S_w'] = calc_headspace_h2s_ppm_from_total_sulfide(
        shs_g_m3=df['SHS'].values,
        Vw_m3=df['Vw'].values,
        Vg_m3=df['Vg'].values,
        pH=PH_H2S, T_K=T_H2S_K, pKa1=PKA1_H2S,
        H_cp=HCP_H2S, MW_S=32.065
    )

    df['SCH4'] = H_A_CH4 * (df['CH4'] * 14 / 32 / 14 * 0.001 / 55.56) * 1e6
    df['rcorr'] = KN_CORR * df['SH2S']**N_CORR / 24

    F_flux = (
        KN_CORR * df['SH2S']**N_CORR *
        df['Vg'] / df['Ag'].replace(0, np.nan) *
        101325 / (R_GAS * T_H2S_K) * 1e-6 * 32
    ).fillna(0)

    df['dcorr_dt'] = CORR_VALUE * 100 / 32 * F_flux / A_CO / CONC_MASS * 24 * 365 * 1000

    R_p = df['diameter'].values / 2
    h = np.clip(df['depth'].values, 0, df['diameter'].values)
    A_tot = np.pi * R_p**2
    sw_, uwp_, Ag_ = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))

    m0 = h == 0
    uwp_[m0] = 2 * np.pi * R_p[m0]
    Ag_[m0] = A_tot[m0]

    mf = h >= df['diameter'].values
    sw_[mf] = 2 * R_p[mf]

    mp = ~(m0 | mf)
    if mp.any():
        th = 2 * np.arccos((R_p[mp] - h[mp]) / np.maximum(R_p[mp], EPS))
        sw_[mp] = 2 * R_p[mp] * np.sin(th / 2)
        Pw = R_p[mp] * th
        uwp_[mp] = 2 * np.pi * R_p[mp] - Pw
        Aw_ = (R_p[mp]**2 * th / 2
               - (R_p[mp] - h[mp]) * np.sqrt(np.maximum(2 * R_p[mp] * h[mp] - h[mp]**2, 0)))
        Ag_[mp] = A_tot[mp] - Aw_

    df['surface_width'] = sw_
    df['unwetted_perimeter'] = uwp_
    df['A_gas'] = Ag_
    df['gas_velocity'] = np.where(uwp_ > 0, 0.682 * sw_ * df['velocity'].values / uwp_, 0)
    df['gas_flow_rate'] = df['gas_velocity'] * Ag_

    df['delta_P'] = 0.5 * RHO_GAS * df['gas_velocity'] ** 2
    df['v_pick']  = C_D_ORIFICE * np.sqrt(2.0 * df['delta_P'] / RHO_GAS)
    df['Q_pick']  = A_PICK * df['v_pick']
    RHO_H2S = RHO_AIR * M_H2S / M_AIR          # ≈ 1.41 kg/m3
    df['H2S_emission_rate'] = df['Q_pick'] * df['SH2S'] * 1e-6 * RHO_H2S   # kg/s

    df['s'] = df['slope'].clip(lower=0)

    # ── 只有 us_is_original / ds_is_original / 起点 is_original 为真时才算 distance ──
    orig_mask = (
        df['us_is_original'].values |
        df['ds_is_original'].values |
        df['node_is_original'].values
    )
    n_eval = int(orig_mask.sum())
    print(f"  distance 仅对原始节点相关 segment 计算: {n_eval:,} / {len(df):,}")

    res_cols = [
        'pipe_name', 'node_name', 'SHS', 'SH2S_w', 'CH4', 'SH2S', 'SCH4',
        'dcorr_dt', 'us_x', 'us_y', 'HRT', 'SO', 'XHw', 'flowrate', 's',
        'velocity', 'surface_width', 'unwetted_perimeter', 'A_gas',
        'gas_velocity', 'gas_flow_rate', 'delta_P', 'v_pick', 'Q_pick',
        'length', 'H2S_emission_rate', 'distance', 'temp_c',
        'us_is_original', 'ds_is_original', 'node_is_original'
    ]

    for c_ppm, label in [(0.1, ""), (0.01, "_odour")]:
        c_tgt = c_ppm * 1e-6 * (RHO_AIR * M_H2S / M_AIR)
        denom = math.pi * U_WIND * SIGMA_Y_A * SIGMA_Z_C * c_tgt

        # 同时满足: 有溢出 + 是原始节点相关 segment
        valid = (df['H2S_emission_rate'].values > 0) & orig_mask

        df['distance'] = np.nan
        df.loc[valid, 'distance'] = (
            df.loc[valid, 'H2S_emission_rate'] / denom
        ) ** (1 / (2 * SIGMA_B))

        out = os.path.join(
            OUT_DIR,
            f"la_emission_radius{label}_segments_v7{suffix}.csv"
        )
        df[res_cols].to_csv(out, index=False, encoding='utf-8-sig')
        print(f"  H2S({c_ppm} ppm) → {out}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Part 8 — Scenario
# ═══════════════════════════════════════════════════════════════════════════════
def run_temp_so4_cod_scenarios_la(
        pipes_split, topo, node_dwf, positive_dwf_nodes,
        scenario_grid="full27"):
    temp_values, so4_values, cod_values = scenario_axes(scenario_grid)

    n_total = len(temp_values) * len(so4_values) * len(cod_values)
    print("\n" + "=" * 70)
    print(f"Scenario: Temperature × SO4 × COD  ({n_total} 组)")
    print(f"  SO4_CV (baseline): {SO4_CV_BASELINE:.4f}  ->  CI = mean × CV")
    print(f"  COD_CV (baseline): {COD_CV_BASELINE:.4f}  ->  CI = mean × CV")
    print("=" * 70)

    run_idx = 0
    scenario_summary = []

    for temp_val in temp_values:
        for so4_val in so4_values:
            for cod_val in cod_values:
                run_idx += 1
                so4_ci_dyn = so4_val * SO4_CV_BASELINE
                cod_ci_dyn = cod_val * COD_CV_BASELINE

                tag = f"T_{temp_val}_SO4_{so4_val}_COD_{cod_val}"
                print(f"\n--- [{run_idx}/{n_total}] {tag} "
                      f"(SO4_CI={so4_ci_dyn:.3f}, COD_CI={cod_ci_dyn:.3f}) ---")

                ci, co, _ = run_simulation_la(
                    pipes_split, topo, node_dwf, positive_dwf_nodes,
                    so4_mean=so4_val, cod_mean=cod_val,
                    so4_ci=so4_ci_dyn, cod_ci=cod_ci_dyn,
                    temp_c=temp_val
                )

                po = postprocess_and_save_la(
                    pipes_split, ci, co, tag=tag, temp_c=temp_val
                )

                emission_df = compute_h2s_emission_la(po, tag=tag, temp_c=temp_val)

                scenario_summary.append({
                    'tag': tag,
                    'temp_c': temp_val,
                    'so4_mean': so4_val,
                    'so4_ci': so4_ci_dyn,
                    'cod_mean': cod_val,
                    'cod_ci': cod_ci_dyn,
                    'mean_SHS_in': po['SHS_in'].mean(),
                    'mean_SHS_out': po['SHS_out'].mean(),
                    'max_SHS_out': po['SHS_out'].max(),
                    'mean_H2S_ppm': emission_df['SH2S'].mean(),
                    'max_H2S_ppm': emission_df['SH2S'].max(),
                    'mean_dcorr_dt': emission_df['dcorr_dt'].mean(),
                    'max_dcorr_dt': emission_df['dcorr_dt'].max(),
                    'total_H2S_emission_rate': emission_df['H2S_emission_rate'].sum()
                })

    summary_df = pd.DataFrame(scenario_summary)
    summary_path = os.path.join(OUT_DIR, "scenario_summary_v7.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\nScenario summary → {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 诊断
# ═══════════════════════════════════════════════════════════════════════════════
def print_basic_diagnostics(pipes_split):
    print("\n" + "=" * 70)
    print("Hydraulic input diagnostics (LA, segment-level)")
    print("=" * 70)

    df = pipes_split
    print(f"\nSegments:")
    print(f"  count              : {len(df):,}")
    print(f"  total length km    : {df['length'].sum() / 1000:.2f}")
    print(f"  total Q cms max    : {df['flowrate'].max():.6f}")
    print(f"  mean Q cms         : {df['flowrate'].mean():.6f}")
    print(f"  nonzero Q count    : {(df['flowrate'] > 0).sum():,}")
    print(f"  mean velocity m/s  : {df['v'].replace(0, np.nan).mean():.4f}")
    print(f"  median depth/D     : "
          f"{(df['depth'] / df['diameter'].replace(0, np.nan)).median():.4f}")
    print(f"  mean HRT hour      : {df['HRT'].replace(0, np.nan).mean():.4f}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 9 — 导出 Shapefile
# ═══════════════════════════════════════════════════════════════════════════════
def export_segments_to_shapefile(pipes_result, emission_df=None,
                                 tag="", crs="EPSG:26911"):
    """
    crs: LA 常用 UTM 11N = "EPSG:26911"；
         若 us_x/us_y 实际为经纬度则用 "EPSG:4326"。
    """
    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except ImportError:
        print("  ⚠ 未安装 geopandas / shapely，无法导出 Shapefile。")
        print("    请先执行: pip install geopandas shapely")
        return None

    suffix = f"_{tag}" if tag else ""
    df = pipes_result.copy()

    need_cols = ['us_x', 'us_y', 'ds_x', 'ds_y']
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        print(f"  ⚠ 缺少端点坐标列 {missing}，无法构建几何，已跳过 Shapefile 导出。")
        return None

    if emission_df is not None:
        em = emission_df.copy()
        em_keep = ['pipe_name', 'SH2S', 'SH2S_w', 'SCH4', 'dcorr_dt',
                   'rcorr', 'H2S_emission_rate', 'distance', 'gas_velocity',
                   'gas_flow_rate', 'A_gas']
        em_keep = [c for c in em_keep if c in em.columns]
        em = em[em_keep].rename(columns={'pipe_name': 'name'})
        em = em.drop_duplicates(subset='name')
        df = df.merge(em, on='name', how='left')

    print("  Building LineString geometries ...")
    geoms = [
        LineString([(ux, uy), (dx, dy)])
        for ux, uy, dx, dy in zip(df['us_x'], df['us_y'],
                                   df['ds_x'], df['ds_y'])
    ]

    valid_mask = []
    for ux, uy, dx, dy in zip(df['us_x'], df['us_y'], df['ds_x'], df['ds_y']):
        ok = not (ux == dx and uy == dy)
        ok = ok and not (ux == 0 and uy == 0)
        ok = ok and not (dx == 0 and dy == 0)
        valid_mask.append(ok)
    valid_mask = np.array(valid_mask)
    n_drop = (~valid_mask).sum()
    if n_drop > 0:
        print(f"    跳过 {n_drop} 条无效几何 (起终点重合或坐标缺失)")

    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs)
    gdf = gdf[valid_mask].reset_index(drop=True)

    base_attrs = ['name', 'start', 'end', 'pipe_type', 'catchment', 'regime',
                  'length', 'diameter', 'slope', 'v', 'depth', 'flowrate',
                  'In_elev', 'Out_elev', 'HRT', 'A_V', 'Vg', 'Vw', 'Ag',
                  'temp_c', 'us_is_original', 'ds_is_original', 'node_is_original']
    conc_attrs = []
    for k in CONC_KEYS:
        conc_attrs += [f'{k}_in', f'{k}_out']
    rate_attrs = ['R5', 'R6', 'R7', 'rs2c', 'rs2b', 'rs2_ox_f']
    emis_attrs = ['SH2S', 'SH2S_w', 'SCH4', 'dcorr_dt', 'rcorr',
                  'H2S_emission_rate', 'distance', 'gas_velocity',
                  'gas_flow_rate', 'A_gas']

    wanted = base_attrs + conc_attrs + rate_attrs + emis_attrs
    wanted = [c for c in wanted if c in gdf.columns]
    keep = wanted + ['geometry']
    gdf = gdf[keep]

    rename_map = {}
    used = set()

    def _short(name, maxlen=10):
        s = name[:maxlen]
        i = 1
        while s in used:
            tail = str(i)
            s = name[:maxlen - len(tail)] + tail
            i += 1
        used.add(s)
        return s

    for col in gdf.columns:
        if col == 'geometry':
            continue
        short = _short(col)
        if short != col:
            rename_map[col] = short
        used.add(col if short == col else short)

    if rename_map:
        map_df = pd.DataFrame(
            [(orig, new) for orig, new in rename_map.items()],
            columns=['original_field', 'shapefile_field']
        )
        map_csv = os.path.join(OUT_DIR, f"la_shapefile_field_map_v7{suffix}.csv")
        map_df.to_csv(map_csv, index=False, encoding='utf-8-sig')
        print(f"  Field-name map (>10 chars) → {map_csv}")
        gdf = gdf.rename(columns=rename_map)

    shp_path = os.path.join(OUT_DIR, f"la_pipes_segments_v7{suffix}.shp")
    gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"  Pipe Shapefile ({len(gdf):,} segments) → {shp_path}")

    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="Run the Los Angeles HRSNM baseline.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-shapefile", action="store_true")
    parser.add_argument(
        "--run-scenarios", action="store_true",
        help="Run the selected temperature/sulphate/COD grid after the baseline.",
    )
    parser.add_argument(
        "--scenario-grid", choices=("small8", "full27"), default="small8",
        help="small8 uses the eight parameter-cube corners; full27 is the manuscript grid.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    OUT_DIR = os.path.abspath(args.out_dir)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("LA Sewer Water Quality Simulation & H2S Analysis")
    print("(segment-level only — NO re-splitting, NO aggregation)")
    print(f"Baseline: T={TEMP}°C, SO4={SO4_MEAN_LA}±{SO4_CI_LA}, "
          f"COD={COD_PER_SOURCE_LA}±{COD_CI_LA}")
    print("=" * 70)
    print(f"\nRUN_SCENARIO_ANALYSIS = {RUN_SCENARIO_ANALYSIS}")

    print("\nStep 1: 读取 segment / node")
    pipes_split_raw, nodes_df = load_segments()

    print("\nStep 2: segment 几何计算")
    pipes_split = compute_geometry(pipes_split_raw)

    print("\nStep 2b: 合并原始节点标志 (us/ds/起点 is_original)")
    pipes_split = attach_original_flags(pipes_split, nodes_df)

    split_csv = os.path.join(OUT_DIR, "la_biochemical_segments_from_hydraulics_v7.csv")
    pipes_split.to_csv(split_csv, index=False, encoding='utf-8-sig')
    print(f"  Segment input snapshot → {split_csv}")

    print_basic_diagnostics(pipes_split)

    print("\nStep 3: 构建拓扑 (segment 级)")
    removed_edges_csv = os.path.join(OUT_DIR, "la_removed_cycle_edges_v7.csv")
    topo = build_topology(pipes_split, removed_edges_csv=removed_edges_csv)

    node_dwf, positive_dwf_nodes = build_node_dwf(nodes_df, topo[0].nodes())
    mixing_summary = node_flow_balance_summary(pipes_split, node_dwf)
    if mixing_summary['n_flow_balance_failures']:
        raise RuntimeError(f"LA node flow-balance audit failed: {mixing_summary}")
    pd.DataFrame([mixing_summary]).to_csv(
        os.path.join(OUT_DIR, "node_dwf_mixing_summary_v7.csv"),
        index=False, encoding='utf-8-sig'
    )
    print(f"  DWF-positive nodes: {mixing_summary['n_positive_dwf_nodes']:,}; "
          f"internal: {mixing_summary['n_internal_positive_dwf_nodes']:,}")

    print("\n" + "=" * 70)
    print(f"Step 4: 基线模拟 (Temp={TEMP} °C)")
    print("=" * 70)

    conc_in_base, conc_out_base, _ = run_simulation_la(
        pipes_split, topo, node_dwf, positive_dwf_nodes,
        so4_mean=SO4_MEAN_LA, cod_mean=COD_PER_SOURCE_LA,
        so4_ci=SO4_CI_LA, cod_ci=COD_CI_LA,
        temp_c=TEMP
    )

    pipes_base = postprocess_and_save_la(
        pipes_split, conc_in_base, conc_out_base, tag="", temp_c=TEMP
    )

    emission_base = compute_h2s_emission_la(pipes_base, tag="", temp_c=TEMP)

    base_summary = {
        'n_segments': len(pipes_split),
        'temp_c': TEMP,
        'so4_mean': SO4_MEAN_LA,
        'so4_ci': SO4_CI_LA,
        'cod_mean': COD_PER_SOURCE_LA,
        'cod_ci': COD_CI_LA,
        'mean_SHS_in': pipes_base['SHS_in'].mean(),
        'mean_SHS_out': pipes_base['SHS_out'].mean(),
        'max_SHS_out': pipes_base['SHS_out'].max(),
        'mean_H2S_ppm': emission_base['SH2S'].mean(),
        'max_H2S_ppm': emission_base['SH2S'].max(),
        'mean_dcorr_dt': emission_base['dcorr_dt'].mean(),
        'max_dcorr_dt': emission_base['dcorr_dt'].max(),
        'total_H2S_emission_rate': emission_base['H2S_emission_rate'].sum()
    }
    pd.DataFrame([base_summary]).to_csv(
        os.path.join(OUT_DIR, "baseline_summary_v7.csv"),
        index=False, encoding='utf-8-sig'
    )

    if args.run_scenarios:
        run_temp_so4_cod_scenarios_la(
            pipes_split, topo, node_dwf, positive_dwf_nodes,
            scenario_grid=args.scenario_grid,
        )
    else:
        print("已跳过 Scenario 分析。如需运行：RUN_SCENARIO_ANALYSIS = True")

    print("\n" + "=" * 70)
    print(f"全部完成 ✓   结果目录: {OUT_DIR}")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Step 5: 导出带管道信息的 Shapefile")
    print("=" * 70)

    if not args.skip_shapefile:
        export_segments_to_shapefile(
            pipes_result=pipes_base,
            emission_df=emission_base,
            tag="",
            crs="EPSG:26911"
        )

    print("\n" + "=" * 70)
    print(f"Shapefile 导出完成 ✓   目录: {OUT_DIR}")
    print("=" * 70)
