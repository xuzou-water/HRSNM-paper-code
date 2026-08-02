# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:39:51 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:01:12 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 18:18:48 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
HK Sewer Network Water Quality Simulation & H2S Emission Analysis
================================================================
直接读取代码2的 segment 级输出,不切分、不聚合。
所有结果直接以 segment 为单位输出。
拓扑(有向图 + 逐层推进)保留——它是浓度沿管网路由的核心。

★ 本版改动:
  计算 H2S 溢出 distance 时,只有当
      us_is_original  或  ds_is_original  或  (节点)is_original
  为真的 segment 才计算 distance,其余 distance = NaN。
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
# 路径
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "processed_data", "hk_v3")
HYD_DIR = os.path.join(DATA_DIR, "hydraulic_results")
DEFAULT_OUT_DIR = os.path.join(
    SCRIPT_DIR, "node_dwf_fix_results", "HK_v3", "biochemical_results"
)
OUT_DIR = DEFAULT_OUT_DIR

# 只需要 segment 水力 + 节点 (不再需要 mapping)
SEG_HYD_CSV = os.environ.get(
    "HRSNM_HK_SEGMENTS_CSV", os.path.join(HYD_DIR, "segment_hydraulics.csv")
)
NODES_ALL_CSV = os.environ.get(
    "HRSNM_HK_NODES_CSV", os.path.join(HYD_DIR, "nodes_all.csv")
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
TEMP = 29.0
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

SO4_MEAN_SEA, SO4_CI = 317.125, 51.95
SO4_MEAN_HK = SO4_MEAN_SEA
SO4_CI_HK   = SO4_CI

COD_PER_SOURCE = 488.0
COD_CI = 52.0
COD_PER_SOURCE_HK = COD_PER_SOURCE
COD_CI_HK = COD_CI

SO4_CV_BASELINE = SO4_CI_HK / SO4_MEAN_HK
COD_CV_BASELINE = COD_CI_HK / COD_PER_SOURCE_HK

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

# ★ 指定 start 节点的 SO_in 强制覆盖值 (HK)
SO_IN_OVERRIDE_HK = {
    'FGJ7008180': 4.56,
    'XPS4001320': 4.28,
    'FYJ4000060': 5.93,
    'XUS4000540': 4.94,
    'FSH4000467': 5.34,
    'FSH7001080': 3.17,
    'XPS4000760': 1.04,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════════════════════════
def _to_bool(series):
    """
    把各种形式的布尔/字符串列统一转换为 numpy bool。
    支持: True/False(已为 bool)、"TRUE"/"FALSE"、"1"/"0"、"T"/"F"、"YES"/"NO"。
    无法识别的一律视为 False。
    """
    if series.dtype == bool:
        return series.astype(bool)
    s = series.astype(str).str.strip().str.upper()
    return s.isin(['TRUE', '1', 'YES', 'T', 'Y']).astype(bool)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1 — 读取代码2的 segment / node  (不再读 mapping, 不再聚合)
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_read_csv(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {name}: {path}")
    return pd.read_csv(path)


def load_segments_from_code2():
    """
    直接读取代码2输出的:
      - segment_hydraulics.csv  每根 segment 的水力 + 几何 + 端点节点
      - nodes_all.csv           所有节点坐标 / invert / type / is_original
    返回:
      seg      : segment 级 DataFrame (即模拟单元)
      nodes_df : 全节点 DataFrame
    """
    print("  Reading segment_hydraulics.csv ...")
    seg = _safe_read_csv(SEG_HYD_CSV, "segment_hydraulics.csv")
    print(f"    segments: {len(seg):,}")

    print("  Reading nodes_all.csv ...")
    nodes_df = _safe_read_csv(NODES_ALL_CSV, "nodes_all.csv")
    nodes_df['node'] = nodes_df['node'].astype(str)
    print(f"    nodes  : {len(nodes_df):,}")

    # 节点 is_original 布尔化
    if 'is_original' in nodes_df.columns:
        nodes_df['is_original'] = _to_bool(nodes_df['is_original'])
    else:
        print("  ⚠ nodes_all.csv 缺少 is_original 列, 全部按 False 处理。")
        nodes_df['is_original'] = False

    # segment 端点 is_original 布尔化
    for c in ['us_is_original', 'ds_is_original']:
        if c in seg.columns:
            seg[c] = _to_bool(seg[c])
        else:
            print(f"  ⚠ segment_hydraulics.csv 缺少 {c} 列, 全部按 False 处理。")
            seg[c] = False

    # 类型规范
    for c in ['link_name', 'parent_link', 'us_node', 'ds_node',
              'pipe_type', 'catchment', 'regime']:
        if c in seg.columns:
            seg[c] = seg[c].astype(str)

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

    return seg, nodes_df


def attach_node_is_original(pipes, nodes_df):
    """
    把节点级 is_original 合并进 segment。
    采用 segment 的起点 (start = us_node) 对应的节点 is_original。
    生成列: node_is_original
    """
    pipes = pipes.copy()
    if 'is_original' in nodes_df.columns:
        node_map = nodes_df.set_index('node')['is_original'].to_dict()
        pipes['node_is_original'] = (
            pipes['start'].astype(str).map(node_map).fillna(False).astype(bool)
        )
    else:
        pipes['node_is_original'] = False

    # 保险: 确保端点 original 列存在
    for c in ['us_is_original', 'ds_is_original']:
        if c not in pipes.columns:
            pipes[c] = False
        else:
            pipes[c] = pipes[c].fillna(False).astype(bool)

    return pipes


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2 — 几何 (A_V, HRT, Vg, Vw, Ag …)
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
# Part 3 — 拓扑 (segment 级)  ★ 路由核心, 必须保留
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
def generate_source_concentrations_hk(source_nodes,
                                      so4_mean=SO4_MEAN_HK,
                                      cod_mean=COD_PER_SOURCE_HK,
                                      so4_ci=SO4_CI_HK,
                                      cod_ci=COD_CI_HK):
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


def run_simulation_hk(pipes_df, topo, node_dwf, positive_dwf_nodes,
                      so4_mean=SO4_MEAN_HK,
                      cod_mean=COD_PER_SOURCE_HK,
                      so4_ci=SO4_CI_HK,
                      cod_ci=COD_CI_HK,
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
    local_inflow_conc = generate_source_concentrations_hk(
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

        # ★ 强制覆盖指定节点的 SO_in (index 3 = 'SO')
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
                forced_do=SO_IN_OVERRIDE_HK.get(nd),
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
# Part 6 — 后处理 & 保存 (★ 直接写 segment, 不聚合)
# ═══════════════════════════════════════════════════════════════════════════════

def postprocess_and_save_hk(pipes_split, conc_in, conc_out, tag="", temp_c=TEMP):
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

    out_path = os.path.join(OUT_DIR, f"hk_result_segments_v7{suffix}.csv")
    p.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  Segment-level WQ results → {out_path}")
    return p
# ═══════════════════════════════════════════════════════════════════════════════
# Part 7 — H2S / CH4 气相、腐蚀、大气扩散  (segment 级)
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


def compute_h2s_emission_hk(pipes_split, tag="", temp_c=TEMP):
    suffix = f"_{tag}" if tag else ""
    fp = pipes_split.copy()

    # ---- 取出 original 标记 (用于决定哪些 segment 计算 distance) ----
    if 'us_is_original' in fp.columns:
        us_orig = _to_bool(fp['us_is_original']).values
    else:
        us_orig = np.zeros(len(fp), dtype=bool)
    if 'ds_is_original' in fp.columns:
        ds_orig = _to_bool(fp['ds_is_original']).values
    else:
        ds_orig = np.zeros(len(fp), dtype=bool)
    if 'node_is_original' in fp.columns:
        nd_orig = _to_bool(fp['node_is_original']).values
    else:
        nd_orig = np.zeros(len(fp), dtype=bool)

    df = pd.DataFrame({
        'pipe_name': fp['name'].values,
        'node_name': fp['start'].values,
        'SHS': fp['SHS_in'].values,
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
        'us_is_original': us_orig,
        'ds_is_original': ds_orig,
        'node_is_original': nd_orig,
    })

    # ★ 计算 distance 的对象掩码:
    #   us_is_original 或 ds_is_original 或 (节点)is_original 为真才计算
    calc_mask = (us_orig | ds_orig | nd_orig)
    df['calc_distance'] = calc_mask
    print(f"  需计算 distance 的 segment 数: {calc_mask.sum():,} / {len(df):,}")

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

    res_cols = [
        'pipe_name', 'node_name', 'SHS', 'SH2S_w', 'CH4', 'SH2S', 'SCH4',
        'dcorr_dt', 'us_x', 'us_y', 'HRT', 'SO', 'XHw', 'flowrate', 's',
        'velocity', 'surface_width', 'unwetted_perimeter', 'A_gas',
        'gas_velocity', 'gas_flow_rate', 'delta_P', 'v_pick', 'Q_pick',
        'length', 'H2S_emission_rate', 'distance', 'temp_c',
        'us_is_original', 'ds_is_original', 'node_is_original', 'calc_distance'
    ]

    for c_ppm, label in [(0.1, ""), (0.01, "_odour")]:
        c_tgt = c_ppm * 1e-6 * (RHO_AIR * M_H2S / M_AIR)
        denom = math.pi * U_WIND * SIGMA_Y_A * SIGMA_Z_C * c_tgt

        # ★ 只在 (排放率>0) 且 (us/ds/node 为 original) 时计算 distance
        valid = (df['H2S_emission_rate'].values > 0) & calc_mask
        df['distance'] = np.nan
        df.loc[valid, 'distance'] = (
            df.loc[valid, 'H2S_emission_rate'] / denom
        ) ** (1 / (2 * SIGMA_B))

        out = os.path.join(
            OUT_DIR,
            f"hk_emission_radius{label}_segments_v7{suffix}.csv"
        )
        df[res_cols].to_csv(out, index=False, encoding='utf-8-sig')
        print(f"  H2S({c_ppm} ppm) → {out}  "
              f"(distance 计算条数: {int(np.isfinite(df['distance']).sum()):,})")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Part 8 — Scenario (segment 级)
# ═══════════════════════════════════════════════════════════════════════════════
def run_temp_so4_cod_scenarios_hk(
        pipes_split, topo, node_dwf, positive_dwf_nodes,
        scenario_grid="full27"):
    temp_values, so4_values, cod_values = scenario_axes(scenario_grid)

    n_total = len(temp_values) * len(so4_values) * len(cod_values)
    print("\n" + "=" * 70)
    print(f"Scenario: Temperature × SO4 × COD  ({n_total} 组)")
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
                print(f"\n--- [{run_idx}/{n_total}] {tag} ---")

                ci, co, _ = run_simulation_hk(
                    pipes_split, topo, node_dwf, positive_dwf_nodes,
                    so4_mean=so4_val, cod_mean=cod_val,
                    so4_ci=so4_ci_dyn, cod_ci=cod_ci_dyn,
                    temp_c=temp_val
                )

                po = postprocess_and_save_hk(
                    pipes_split, ci, co, tag=tag, temp_c=temp_val
                )

                emission_df = compute_h2s_emission_hk(po, tag=tag, temp_c=temp_val)

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
    summary_df.to_csv(os.path.join(OUT_DIR, "scenario_summary_v7.csv"),
                      index=False, encoding='utf-8-sig')


# ═══════════════════════════════════════════════════════════════════════════════
# Part 9 — 导出带管道信息的 Shapefile (segment 级)
# ═══════════════════════════════════════════════════════════════════════════════
def export_segments_to_shapefile(pipes_result, emission_df=None,
                                 tag="", crs="EPSG:2326"):
    """
    将 segment 级结果导出为 Shapefile。
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
        map_csv = os.path.join(OUT_DIR, f"hk_shapefile_field_map_v7{suffix}.csv")
        map_df.to_csv(map_csv, index=False, encoding='utf-8-sig')
        print(f"  Field-name map (>10 chars) → {map_csv}")
        gdf = gdf.rename(columns=rename_map)

    shp_path = os.path.join(OUT_DIR, f"hk_pipes_segments_v7{suffix}.shp")
    gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"  Pipe Shapefile ({len(gdf):,} segments) → {shp_path}")

    return gdf


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="Run the Hong Kong HRSNM baseline.")
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


def main():
    global OUT_DIR
    args = parse_args()
    OUT_DIR = os.path.abspath(args.out_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("HK Sewer Water Quality Simulation & H2S Analysis")
    print("(segment-level only — NO re-splitting, NO aggregation)")
    print("=" * 70)
    print(f"\nRUN_SCENARIO_ANALYSIS = {RUN_SCENARIO_ANALYSIS}")

    print("\nStep 1: 读取代码2的 segment / node")
    pipes_split_raw, nodes_df = load_segments_from_code2()

    print("\nStep 2: segment 几何计算")
    pipes_split = compute_geometry(pipes_split_raw)

    print("\nStep 2b: 合并节点 is_original (按起点 us_node)")
    pipes_split = attach_node_is_original(pipes_split, nodes_df)
    print(f"  us_is_original=True  : {int(pipes_split['us_is_original'].sum()):,}")
    print(f"  ds_is_original=True  : {int(pipes_split['ds_is_original'].sum()):,}")
    print(f"  node_is_original=True: {int(pipes_split['node_is_original'].sum()):,}")

    split_csv = os.path.join(OUT_DIR, "hk_biochemical_segments_from_hydraulics_v7.csv")
    pipes_split.to_csv(split_csv, index=False, encoding='utf-8-sig')
    print(f"  Segment input snapshot → {split_csv}")

    print("\nStep 3: 构建拓扑 (segment 级)")
    removed_edges_csv = os.path.join(OUT_DIR, "hk_removed_cycle_edges_v7.csv")
    topo = build_topology(pipes_split, removed_edges_csv=removed_edges_csv)

    node_dwf, positive_dwf_nodes = build_node_dwf(nodes_df, topo[0].nodes())
    mixing_summary = node_flow_balance_summary(pipes_split, node_dwf)
    if mixing_summary['n_flow_balance_failures']:
        raise RuntimeError(f"HK node flow-balance audit failed: {mixing_summary}")
    pd.DataFrame([mixing_summary]).to_csv(
        os.path.join(OUT_DIR, "node_dwf_mixing_summary_v7.csv"),
        index=False, encoding='utf-8-sig'
    )
    print(f"  DWF-positive nodes: {mixing_summary['n_positive_dwf_nodes']:,}; "
          f"internal: {mixing_summary['n_internal_positive_dwf_nodes']:,}")

    print("\n" + "=" * 70)
    print(f"Step 4: 基线模拟 (Temp={TEMP} °C)")
    print("=" * 70)

    conc_in_base, conc_out_base, _ = run_simulation_hk(
        pipes_split, topo, node_dwf, positive_dwf_nodes,
        so4_mean=SO4_MEAN_HK, cod_mean=COD_PER_SOURCE_HK,
        so4_ci=SO4_CI_HK, cod_ci=COD_CI_HK,
        temp_c=TEMP
    )

    pipes_base = postprocess_and_save_hk(
        pipes_split, conc_in_base, conc_out_base, tag="", temp_c=TEMP
    )

    emission_base = compute_h2s_emission_hk(pipes_base, tag="", temp_c=TEMP)

    base_summary = {
        'n_segments': len(pipes_split),
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
        run_temp_so4_cod_scenarios_hk(
            pipes_split, topo, node_dwf, positive_dwf_nodes,
            scenario_grid=args.scenario_grid,
        )
    else:
        print("已跳过 Scenario 分析。如需运行:RUN_SCENARIO_ANALYSIS = True")

    print("\n" + "=" * 70)
    print(f"全部完成 ✓   结果目录: {OUT_DIR}")
    print("=" * 70)

    # ---- Step 5: 导出 Shapefile ----
    print("\n" + "=" * 70)
    print("Step 5: 导出带管道信息的 Shapefile")
    print("=" * 70)
    if not args.skip_shapefile:
        export_segments_to_shapefile(
            pipes_result=pipes_base,
            emission_df=emission_base,
            tag="",
            crs="EPSG:2326"
        )
    print("\n" + "=" * 70)
    print(f"Shapefile 导出完成 ✓   目录: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
# #%
# # ═══════════════════════════════════════════════════════════════════════════════
# # Part 10 — 绘制代码2的最后一个子图 (TDS 验证: Predicted vs Measured)
# # ═══════════════════════════════════════════════════════════════════════════════
# def plot_tds_validation():
#     """
#     复现代码2中的最后一个子图 (ax_tval):
#     按 node (pipes 的 'start' 列) 查找预测 SHS_in,
#     与 measurement_TDS_update5.csv 的测量 TDS 对比, 画分组柱状图。
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from scipy import stats

#     # ---------- 色系 ----------
#     SPINE_C   = "#333333"
#     TEXT_C    = "#222222"
#     GRID_C    = "#D9D4CB"
#     SAGE      = "#9DBEBA"   # prediction
#     BEIGE     = "#9B8F7A"   # measurement
#     COLOR_PRED, COLOR_MEAS = SAGE, BEIGE
#     BG_FIG    = "white"
#     BG_AX     = "white"

#     try:
#         plt.rcParams["font.family"] = "Arial"
#     except Exception:
#         plt.rcParams["font.family"] = "sans-serif"
#     plt.rcParams.update({
#         "axes.unicode_minus": False,
#         "axes.edgecolor":  SPINE_C,
#         "axes.labelcolor": TEXT_C,
#         "xtick.color":     TEXT_C,
#         "ytick.color":     TEXT_C,
#         "text.color":      TEXT_C,
#         "axes.linewidth":  0.7,
#         "xtick.direction": "in",
#         "ytick.direction": "in",
#     })

#     # ---------- 路径 ----------
#     result_path   = os.path.join(OUT_DIR, "hk_result_segments_v7.csv")
#     tds_meas_path = os.path.join(
#         SCRIPT_DIR, "data", "figure1", "measurement_TDS_update5.csv"
#     )

#     if not os.path.exists(result_path):
#         print(f"  ⚠ 找不到结果文件: {result_path}")
#         return
#     if not os.path.exists(tds_meas_path):
#         print(f"  ⚠ 找不到测量文件: {tds_meas_path}")
#         return

#     # ---------- 读取预测结果 ----------
#     pipes = pd.read_csv(result_path)
#     pipes['_start_str_lookup'] = pipes['start'].astype(str).str.strip()

#     # ---------- 测量数据读取 ----------
#     def calc_95ci_half_width(series):
#         s = series.dropna()
#         n = len(s)
#         if n < 2:
#             return 0.0
#         return stats.t.ppf(0.975, df=n - 1) * s.sem()

#     mp_labels = ['MP1', 'MP2', 'MP3', 'MP4', 'MP5', 'MP6', 'MP7', 'MP8']

#     meas_raw  = pd.read_csv(tds_meas_path, sep=None, engine='python', header=0)
#     node_row  = meas_raw.iloc[0]
#     meas_data = meas_raw.iloc[1:].reset_index(drop=True)

#     cols      = meas_raw.columns.tolist()
#     tds_cols  = cols[1:9]     # 第2~9列 -> TDS (8 列)

#     assert len(tds_cols) == len(mp_labels), \
#         f"TDS 列数 ({len(tds_cols)}) 与 MP 数 ({len(mp_labels)}) 不匹配"

#     for c in tds_cols:
#         meas_data[c] = pd.to_numeric(meas_data[c], errors='coerce')

#     # MP -> node 名称映射
#     node_names = [str(node_row[c]).strip() for c in tds_cols]
#     mp_to_node = dict(zip(mp_labels, node_names))
#     print('\n  MP -> node 映射:')
#     for mp in mp_labels:
#         print(f'    {mp}: {mp_to_node[mp]}')

#     # 测量统计
#     tds_means  = {mp: meas_data[c].dropna().mean()       for mp, c in zip(mp_labels, tds_cols)}
#     tds_errors = {mp: calc_95ci_half_width(meas_data[c]) for mp, c in zip(mp_labels, tds_cols)}

#     # ---------- 用 node 查找预测 SHS_in ----------
#     pred_tds = {}
#     for mp in mp_labels:
#         node  = mp_to_node[mp]
#         match = pipes[pipes['_start_str_lookup'] == node]
#         if len(match) > 0:
#             pred_tds[mp] = match.iloc[0]['SHS_in']
#         else:
#             pred_tds[mp] = np.nan
#             print(f'  ⚠ 未在 pipes.start 中找到 node {node} (对应 {mp})')

#     display_order = mp_labels

#     # ---------- 绘图用 DataFrame ----------
#     filtered_df = pd.DataFrame({
#         'display_name':             mp_labels,
#         'SHS_in':                   [pred_tds[mp]   for mp in mp_labels],
#         'SHS_in_measurement_mean':  [tds_means[mp]  for mp in mp_labels],
#         'SHS_in_measurement_error': [tds_errors[mp] for mp in mp_labels],
#     })

#     plot_df_tds = filtered_df.melt(
#         id_vars='display_name',
#         value_vars=['SHS_in', 'SHS_in_measurement_mean'],
#         var_name='concentration_type',
#         value_name='concentration'
#     )

#     # ---------- 画图 ----------
#     SPINE_WIDTH_RIGHT = 0.7
#     ERR_LINE_WIDTH    = 0.7
#     fs_label, fs_tick = 11, 9

#     fig, ax_tval = plt.subplots(figsize=(14 / 2, 5 / 2), dpi=300)
#     fig.patch.set_facecolor(BG_FIG)

#     ax_tval.set_facecolor(BG_AX)
#     ax_tval.spines['top'].set_visible(False)
#     ax_tval.spines['right'].set_visible(False)
#     ax_tval.spines['left'].set_linewidth(SPINE_WIDTH_RIGHT)
#     ax_tval.spines['bottom'].set_linewidth(SPINE_WIDTH_RIGHT)
#     ax_tval.spines['left'].set_edgecolor(SPINE_C)
#     ax_tval.spines['bottom'].set_edgecolor(SPINE_C)
#     ax_tval.tick_params(direction='in', width=SPINE_WIDTH_RIGHT, colors=TEXT_C)
#     ax_tval.grid(True, axis='y', linestyle=':', linewidth=0.5, color=GRID_C, alpha=0.45)
#     ax_tval.set_axisbelow(True)

#     palette_tds = {'SHS_in': COLOR_PRED, 'SHS_in_measurement_mean': COLOR_MEAS}
#     sns.barplot(
#         data=plot_df_tds, x='display_name', y='concentration',
#         hue='concentration_type',
#         hue_order=['SHS_in', 'SHS_in_measurement_mean'],
#         palette=palette_tds, order=display_order,
#         linewidth=0, edgecolor='none',
#         ax=ax_tval, errorbar=None
#     )

#     # 误差棒 (只加在测量柱上)
#     patches_g = ax_tval.patches
#     n_g_g = len(display_order)
#     for i, dname in enumerate(display_order):
#         bar = patches_g[n_g_g + i]
#         bx = bar.get_x() + bar.get_width() / 2
#         by = bar.get_height()
#         err_val = filtered_df.loc[
#             filtered_df['display_name'] == dname, 'SHS_in_measurement_error'
#         ].values[0]
#         if pd.isna(err_val):
#             err_val = 0.0
#         ax_tval.errorbar(bx, by, yerr=err_val, fmt='none', ecolor=SPINE_C,
#                          capsize=2.0, capthick=ERR_LINE_WIDTH, linewidth=ERR_LINE_WIDTH)

#     handles_g, labels_g = ax_tval.get_legend_handles_labels()
#     labels_g = ['Predicted TDS' if l == 'SHS_in'
#                 else 'Measured TDS' if l == 'SHS_in_measurement_mean'
#                 else l for l in labels_g]
#     leg_t = ax_tval.legend(
#         handles=handles_g, labels=labels_g,
#         fontsize=fs_tick, loc='upper right', frameon=False
#     )
#     for t in leg_t.get_texts():
#         t.set_color(TEXT_C)

#     ax_tval.set_xlabel('Measurement Points', fontsize=fs_label, color=TEXT_C)
#     ax_tval.set_ylabel('TDS [gS/m³]', fontsize=fs_label, color=TEXT_C)
#     ax_tval.tick_params(labelsize=fs_tick, width=SPINE_WIDTH_RIGHT, colors=TEXT_C)

#     plt.tight_layout()

#     out_png = os.path.join(OUT_DIR, "Fig_TDS_validation.png")
#     plt.savefig(out_png, dpi=600, bbox_inches='tight',
#                 facecolor=BG_FIG, transparent=False)
#     print(f"\n  TDS 验证图 → {out_png}")
#     plt.show()


# if __name__ == "__main__":
#     main()
#     # ---- 画代码2的最后一个子图 (TDS 验证) ----
#     print("\n" + "=" * 70)
#     print("Step 6: 绘制 TDS 验证图 (代码2最后一个子图)")
#     print("=" * 70)
#     plot_tds_validation()
