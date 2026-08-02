# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:18:32 2026

@author: zouxu
"""

# -*- coding: utf-8 -*-
"""
Modified Combined Figure — node-based SHS_in lookup (6 measurement points).

数据结构（共 15 列，6 个测量点）：
   表头第1行: Name | MP4 MP3 MP2 MP1 MP6 MP5 | (空) | Name | MP4 MP3 MP2 MP1 MP6 MP5
   第2行     : node_name | FYJ... XUS... ... | (空) | date | FYJ... XUS... ...
   第3行起   : 时间戳 + 数值
   - 第2~7列   = TDS 浓度    (6 列)
   - 第8列     = 空分隔列 (跳过)
   - 第9列     = 第二个 Name/date 表头列
   - 第10~15列 = 流量 (m3/d) (6 列)
"""

import argparse
import os
import re
import warnings
from collections import deque

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import geopandas as gpd
import seaborn as sns
from scipy import stats
from scipy.spatial import cKDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_parser = argparse.ArgumentParser(description="Generate HRSNM Figure 1.")
_parser.add_argument(
    "--result-csv",
    default=os.path.join(
        SCRIPT_DIR, "node_dwf_fix_results", "HK_v3",
        "biochemical_results", "hk_result_segments_v7.csv",
    ),
)
_parser.add_argument(
    "--output-dir",
    default=os.path.join(SCRIPT_DIR, "node_dwf_fix_results", "figure1"),
)
_parser.add_argument(
    "--building-shp",
    default=os.environ.get(
        "HRSNM_FIG1_BUILDING_SHP",
        os.path.join(SCRIPT_DIR, "data", "figure1", "HK_buildings.shp"),
    ),
)
_parser.add_argument(
    "--sewer-csv",
    default=os.environ.get(
        "HRSNM_FIG1_SEWER_CSV",
        os.path.join(SCRIPT_DIR, "data", "figure1", "sewer_pipes_filled.csv"),
    ),
)
_parser.add_argument(
    "--total-hrt-csv",
    default=os.environ.get(
        "HRSNM_FIG1_TOTAL_HRT_CSV",
        os.path.join(SCRIPT_DIR, "data", "figure1", "total_hrt_per_node.csv"),
    ),
)
_parser.add_argument(
    "--measurement-csv",
    default=os.environ.get(
        "HRSNM_FIG1_MEASUREMENT_CSV",
        os.path.join(SCRIPT_DIR, "data", "figure1", "measurement_TDS_update6.csv"),
    ),
)
_parser.add_argument("--no-show", action="store_true")
FIG_ARGS = _parser.parse_args()

# =============================================================================
# 0. 统一色系
# =============================================================================
BG_FIG       = "white"
BG_AX        = "white"
BG_MAP       = "#F7F6F2"
GRID_C       = "#D9D4CB"
SPINE_C      = "#333333"
TEXT_C       = "#222222"
SUBTEXT_C    = "#666666"

SAGE_LIGHT   = "#9DBEBA"
SAGE         = "#9DBEBA"  # prediction
SAGE_DARK    = "#82BFB0"
TEAL         = "#3A7A8C"
NAVY         = "#08345A"
BEIGE        = "#9B8F7A"  # measurement
BROWN        = "#A66842"
ACCENT_RED   = "#9C3106"

try:
    plt.rcParams["font.family"] = "Arial"
except Exception:
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams.update({
    "axes.unicode_minus": False,
    "axes.edgecolor":     SPINE_C,
    "axes.labelcolor":    TEXT_C,
    "xtick.color":        TEXT_C,
    "ytick.color":        TEXT_C,
    "text.color":         TEXT_C,
    "axes.linewidth":     0.7,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# =============================================================================
# 1. 读取数据
# =============================================================================
result_path = os.path.abspath(FIG_ARGS.result_csv)
building_path = os.path.abspath(FIG_ARGS.building_shp)
sewer_pipes_filled_path = os.path.abspath(FIG_ARGS.sewer_csv)
total_hrt_per_node_path = os.path.abspath(FIG_ARGS.total_hrt_csv)
tds_meas_path = os.path.abspath(FIG_ARGS.measurement_csv)

pipes = pd.read_csv(result_path)

try:
    buildings = gpd.read_file(building_path, encoding='big5')
except UnicodeDecodeError:
    try:
        buildings = gpd.read_file(building_path, encoding='cp950')
    except UnicodeDecodeError:
        buildings = gpd.read_file(building_path, encoding='latin1')
        print('⚠ 使用 latin1 编码读取')

sewer_pipes_df = pd.read_csv(sewer_pipes_filled_path)
pipes_df       = pd.read_csv(result_path)
total_hrt_df   = pd.read_csv(total_hrt_per_node_path)

print(f'管道数量: {len(pipes)}')
print(f'建筑物数量: {len(buildings)}')

# =============================================================================
# 1.5 排除 Link_, 构建有向图，筛选主要 end node
# =============================================================================
pipes_no_link = pipes[~pipes['name'].astype(str).str.startswith('Link_')].copy()

valid_mask  = pipes_no_link['start'].notna() & pipes_no_link['end'].notna()
valid_pipes = pipes_no_link[valid_mask]
starts_arr  = valid_pipes['start'].astype(str).values
ends_arr    = valid_pipes['end'].astype(str).values

forward_adj, reverse_adj = {}, {}
for s, e in zip(starts_arr, ends_arr):
    forward_adj.setdefault(s, []).append(e)
    reverse_adj.setdefault(e, []).append(s)
    forward_adj.setdefault(e, [])
    reverse_adj.setdefault(s, [])

all_graph_nodes = set(forward_adj.keys()) | set(reverse_adj.keys())
sink_nodes = [n for n in all_graph_nodes if len(forward_adj.get(n, [])) == 0]

node_to_sink = {}
bfs_queue = deque()
for sink in sink_nodes:
    node_to_sink[sink] = sink
    bfs_queue.append(sink)

while bfs_queue:
    current = bfs_queue.popleft()
    for upstream in reverse_adj.get(current, []):
        if upstream not in node_to_sink:
            node_to_sink[upstream] = node_to_sink[current]
            bfs_queue.append(upstream)

pipes_no_link['end_node'] = pipes_no_link['start'].astype(str).map(node_to_sink)
sink_counts    = pipes_no_link['end_node'].value_counts()
dominant_sink  = sink_counts.idxmax()
pipes_dominant = pipes_no_link[pipes_no_link['end_node'] == dominant_sink].copy()
print(f'保留管道: {len(pipes_dominant)} / {len(pipes_no_link)}')

pipes_with_coords = pipes_dominant.dropna(subset=['us_x', 'us_y', 'ds_x', 'ds_y'])
if len(pipes_with_coords) > 0:
    pipe_pts = np.concatenate([
        pipes_with_coords[['us_x', 'us_y']].values,
        pipes_with_coords[['ds_x', 'ds_y']].values
    ])
    pipe_tree = cKDTree(pipe_pts)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bldg_x = buildings.geometry.centroid.x.values
        bldg_y = buildings.geometry.centroid.y.values

    bldg_xy  = np.column_stack([bldg_x, bldg_y])
    dists, _ = pipe_tree.query(bldg_xy)
    buildings_filtered = buildings[dists <= 1000].copy()
else:
    buildings_filtered = buildings.iloc[0:0].copy()

# =============================================================================
# 2. 地图数据
# =============================================================================
pipes_map = pipes_dominant.dropna(subset=['us_x', 'us_y', 'ds_x', 'ds_y']).reset_index(drop=True)

def classify_shs(val):
    if val <= 0.5:   return 0
    elif val <= 2.0: return 1
    else:            return 2

pipes_map['shs_class'] = pipes_map['SHS_out'].apply(classify_shs)

color_map_shs = {0: SAGE_DARK, 1: TEAL, 2: NAVY}
label_map     = {0: 'TDS ≤ 0.5 g/m³', 1: '0.5 < TDS ≤ 2 g/m³', 2: 'TDS > 2 g/m³'}

pipes_sorted = pipes_map.sort_values('shs_class', ascending=True).reset_index(drop=True)
us_xy      = pipes_sorted[['us_x', 'us_y']].values / 1000
ds_xy      = pipes_sorted[['ds_x', 'ds_y']].values / 1000
segments   = np.stack([us_xy, ds_xy], axis=1)
colors_seg = pipes_sorted['shs_class'].map(color_map_shs).tolist()

x_all = np.concatenate([pipes_map['us_x'].values, pipes_map['ds_x'].values]) / 1000
y_all = np.concatenate([pipes_map['us_y'].values, pipes_map['ds_y'].values]) / 1000
map_pad, map_pad_y = 0.5, 0.5
MAIN_XLIM = (x_all.min() - map_pad,   x_all.max() + map_pad)
MAIN_YLIM = (y_all.min() - map_pad_y, y_all.max() + map_pad_y)

# =============================================================================
# 3. 管道属性分布
# =============================================================================
if 'FR_PNT' in sewer_pipes_df.columns:
    sewer_pipes_df.rename(columns={'FR_PNT': 'Node_Name'}, inplace=True)

pipes_dominant['_start_str'] = pipes_dominant['start'].astype(str)
sewer_pipes_df['Node_Name']  = sewer_pipes_df['Node_Name'].astype(str)

pipes_dist = pd.merge(
    pipes_dominant, sewer_pipes_df,
    left_on='_start_str', right_on='Node_Name',
    how='left', suffixes=('', '_sewer')
)

def get_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            print(f"  ✓ {label}: '{c}'")
            return c
    print(f"  ⚠ {label}: 未找到")
    return None

def safe_extract(df, col):
    if col is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors='coerce').dropna()

col_av       = get_col(pipes_dist, ['A_V', 'a_v', 'AV', 'av'], 'A/V')
col_velocity = get_col(pipes_dist, ['v', 'V', 'velocity', 'Velocity', 'VELOCITY', 'v_sewer'], 'Velocity')
col_flowrate = get_col(pipes_dist, ['flowrate', 'Flowrate', 'FLOWRATE', 'flow', 'Q'], 'Flowrate')
col_diameter = get_col(pipes_dist, ['WIDTH', 'width', 'diameter', 'DIAMETER', 'DIAM', 'diam',
                                    'width_sewer', 'WIDTH_sewer', 'diameter_sewer', 'DIAMETER_sewer'], 'Diameter')
col_slope    = get_col(pipes_dist, ['slope', 'SLOPE', 'gradient', 'GRADIENT', 'grad',
                                    'slope_sewer', 'SLOPE_sewer'], 'Slope')

data_av       = safe_extract(pipes_dist, col_av)
data_velocity = safe_extract(pipes_dist, col_velocity)
data_flowrate = safe_extract(pipes_dist, col_flowrate)
data_diameter = safe_extract(pipes_dist, col_diameter)
data_slope    = safe_extract(pipes_dist, col_slope)

depth_col = 'depth' if 'depth' in pipes_dist.columns else None
diam_col  = 'diameter' if 'diameter' in pipes_dist.columns else col_diameter
if depth_col is not None and diam_col is not None:
    _depth = pd.to_numeric(pipes_dist[depth_col], errors='coerce')
    _diam  = pd.to_numeric(pipes_dist[diam_col],  errors='coerce')
    _filling_series = _depth / _diam
    _filling_series = _filling_series.replace([np.inf, -np.inf], np.nan)
    _valid_mask = _filling_series.notna() & (_filling_series > 0) & (_filling_series <= 1.5)
    data_filling = _filling_series[_valid_mask]
else:
    data_filling = pd.Series(dtype=float)

# =============================================================================
# 计算并导出每个参数的 whisker range
# =============================================================================
def remove_outliers_iqr(data, k=1.5):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return data
    q1  = np.percentile(data, 25)
    q3  = np.percentile(data, 75)
    iqr = q3 - q1
    lo  = q1 - k * iqr
    hi  = q3 + k * iqr
    mask = (data >= lo) & (data <= hi)
    return data[mask]

def compute_whisker_range(data, k=1.5):
    arr = pd.to_numeric(data, errors='coerce').dropna().values
    if len(arr) < 1:
        return np.nan, np.nan
    clean = remove_outliers_iqr(arr, k=k)
    if len(clean) < 1:
        clean = arr
    return float(np.min(clean)), float(np.max(clean))

whisker_ranges = {}
for lbl, d in [('A/V',           data_av),
               ('Velocity',      data_velocity),
               ('Flowrate',      data_flowrate),
               ('Pipe diameter', data_diameter),
               ('Slope',         data_slope),
               ('Filling ratio', data_filling)]:
    wmin, wmax = compute_whisker_range(d, k=1.5)
    whisker_ranges[lbl] = (wmin, wmax)
    print(f"  Whisker range [{lbl}] = [{wmin:.4g}, {wmax:.4g}]")

save_dir = os.path.abspath(FIG_ARGS.output_dir)
os.makedirs(save_dir, exist_ok=True)

whisker_df = pd.DataFrame([
    {'parameter': k, 'whisker_min': v[0], 'whisker_max': v[1]}
    for k, v in whisker_ranges.items()
])
whisker_csv_path = os.path.join(save_dir, 'hk_whisker_ranges.csv')
whisker_df.to_csv(whisker_csv_path, index=False)
print(f"✓ HK whisker ranges 已保存至: {whisker_csv_path}")

# =============================================================================
# 4. TDS / Flowrate 测量数据读取 + 基于 node 的预测查找（6 个测量点）
# =============================================================================
def calc_95ci_half_width(series):
    s = series.dropna()
    n = len(s)
    if n < 2:
        return 0.0
    return stats.t.ppf(0.975, df=n - 1) * s.sem()

def clean_tds_value(x):
    """清洗 TDS 值：处理 '＜0.1' / '<0.1' / '＞5' / '>5' 等带（全/半角）小于、大于号的值。"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == '' or s.lower() in ('nan', 'na', 'none'):
        return np.nan
    for ch in ['＜', '<', '＞', '>', '≤', '≥', '≦', '≧']:
        s = s.replace(ch, '')
    s = s.replace(' ', '').replace('，', '').replace(',', '')
    try:
        return float(s)
    except ValueError:
        return np.nan

# ---- 读取测量文件（自动检测分隔符：制表符/逗号）----
meas_raw = pd.read_csv(tds_meas_path, sep=None, engine='python', header=0)

# 第2行（index 0）为 node_name 行，其余为时间序列数据
node_row  = meas_raw.iloc[0]
meas_data = meas_raw.iloc[1:].reset_index(drop=True)

cols = meas_raw.columns.tolist()

# 第2~7列 = TDS (6 列)；最后 6 列 = Flowrate (自动跳过中间空列与第二个表头列)
tds_cols  = cols[1:7]
flow_cols = cols[-6:]

# 测量点标签直接取表头（保留真实顺序与名称：MP4, MP3, MP2, MP1, MP6, MP5）
mp_labels = [str(c).strip() for c in tds_cols]

# 安全检查：确保取到 6 列
assert len(tds_cols)  == 6, f"TDS 列数 ({len(tds_cols)}) 不等于 6"
assert len(flow_cols) == 6, f"Flowrate 列数 ({len(flow_cols)}) 不等于 6"

# 数值化（TDS 用专用清洗函数处理 '＜0.1' 等；Flowrate 直接转数值）
for c in tds_cols:
    meas_data[c] = meas_data[c].apply(clean_tds_value)
for c in flow_cols:
    meas_data[c] = pd.to_numeric(meas_data[c], errors='coerce')

# MP -> node 名称映射（TDS 与 Flowrate 的 node 相同，用 TDS 列即可）
node_names = [str(node_row[c]).strip() for c in tds_cols]
mp_to_node = dict(zip(mp_labels, node_names))
print('\nMP -> node 映射:')
for mp in mp_labels:
    print(f'  {mp}: {mp_to_node[mp]}')

# 测量统计
tds_means  = {mp: meas_data[c].dropna().mean()        for mp, c in zip(mp_labels, tds_cols)}
tds_errors = {mp: calc_95ci_half_width(meas_data[c])  for mp, c in zip(mp_labels, tds_cols)}
flow_means = {mp: meas_data[c].dropna().mean()        for mp, c in zip(mp_labels, flow_cols)}
flow_stds  = {mp: meas_data[c].dropna().std()         for mp, c in zip(mp_labels, flow_cols)}

# ---- 用 node 在 pipes 的 'start' 列查找预测 SHS_in / flowrate ----
pipes['_start_str_lookup'] = pipes['start'].astype(str).str.strip()

pred_tds, pred_flow = {}, {}
for mp in mp_labels:
    node  = mp_to_node[mp]
    match = pipes[pipes['_start_str_lookup'] == node]
    if len(match) > 0:
        pred_tds[mp]  = match.iloc[0]['SHS_in']
        pred_flow[mp] = match.iloc[0]['flowrate'] * 86400  # m3/s -> m3/d
    else:
        pred_tds[mp]  = np.nan
        pred_flow[mp] = np.nan
        print(f'⚠ 未在 pipes.start 中找到 node {node} (对应 {mp})')

# ---- 显示顺序：按 MP 编号升序排列 (MP1, MP2, ..., MP6) ----
def mp_sort_key(label):
    m = re.search(r'\d+', str(label))
    return int(m.group()) if m else 0

display_order = sorted(mp_labels, key=mp_sort_key)
print(f'\n显示顺序: {display_order}')

# ---- TDS 绘图用 DataFrame ----
filtered_df = pd.DataFrame({
    'display_name':              mp_labels,
    'SHS_in':                    [pred_tds[mp]  for mp in mp_labels],
    'SHS_in_measurement_mean':   [tds_means[mp] for mp in mp_labels],
    'SHS_in_measurement_error':  [tds_errors[mp] for mp in mp_labels],
})

plot_df_tds = filtered_df.melt(
    id_vars='display_name',
    value_vars=['SHS_in', 'SHS_in_measurement_mean'],
    var_name='concentration_type',
    value_name='concentration'
)

# =============================================================================
# 5. Flowrate 验证数据
# =============================================================================
comparison_df = pd.DataFrame({
    'Plot_Name':                mp_labels,
    'Measured_Flowrate_m3d':    [flow_means[mp] for mp in mp_labels],
    'Measured_Flowrate_std':    [flow_stds[mp]  for mp in mp_labels],
    'Validation_Flowrate_m3d':  [pred_flow[mp]  for mp in mp_labels],
})

melted_comparison_df = comparison_df.melt(
    id_vars=['Plot_Name'],
    value_vars=['Validation_Flowrate_m3d', 'Measured_Flowrate_m3d'],
    var_name='Flowrate_Type',
    value_name='Flowrate_m3d'
)
melted_comparison_df['Flowrate_Type'] = melted_comparison_df['Flowrate_Type'].replace({
    'Measured_Flowrate_m3d':   'Measured flowrate',
    'Validation_Flowrate_m3d': 'Predicted flowrate'
})
melted_comparison_df['Flowrate_m3d'] /= 1e6
std_dict = dict(zip(comparison_df['Plot_Name'], comparison_df['Measured_Flowrate_std'] / 1e6))

# =============================================================================
# 6. 组合绘图
# =============================================================================
fig_w, fig_h = 28 / 2, 22 / 2
fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
fig.patch.set_facecolor(BG_FIG)

gs = GridSpec(
    150, 138, figure=fig, hspace=35, wspace=20,
    width_ratios=[1.5] * 70 + [1.0] * 68
)

ax_map  = fig.add_subplot(gs[35:147,  0:75])
ax_av   = fig.add_subplot(gs[0:25,   83:107])
ax_vel  = fig.add_subplot(gs[0:25,  114:138])
ax_flow = fig.add_subplot(gs[30:55,  83:107])
ax_diam = fig.add_subplot(gs[30:55, 114:138])
ax_slp  = fig.add_subplot(gs[60:85,  83:107])
ax_full = fig.add_subplot(gs[60:85, 114:138])
ax_fval = fig.add_subplot(gs[90:115,  83:138])
ax_tval = fig.add_subplot(gs[122:147, 83:138])

SPINE_WIDTH_MAP, SPINE_WIDTH_RIGHT = 0.7, 0.7
BOX_LINEWIDTH, MEDIAN_WIDTH = 0.65, 1.1
WHISKER_WIDTH, CAP_WIDTH    = 0.7, 0.7
BAR_EDGE_WIDTH, ERR_LINE_WIDTH, LEGEND_EDGE_WIDTH = 0.65, 0.7, 0.6
fs_label, fs_tick = 11, 9

BOX_COLOR, BOX_EDGE = SAGE_LIGHT, SPINE_C
MEDIAN_COLOR, MEAN_COLOR = ACCENT_RED, "#375350"
COLOR_PRED, COLOR_MEAS   = SAGE, BEIGE

def apply_half_frame(ax, spine_width=0.7):
    ax.set_facecolor(BG_AX)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_linewidth(spine_width)
    ax.spines['bottom'].set_linewidth(spine_width)
    ax.spines['left'].set_edgecolor(SPINE_C)
    ax.spines['bottom'].set_edgecolor(SPINE_C)
    ax.tick_params(direction='in', width=spine_width, colors=TEXT_C)

for ax in [ax_av, ax_vel, ax_flow, ax_diam, ax_slp, ax_full, ax_fval, ax_tval]:
    apply_half_frame(ax, spine_width=SPINE_WIDTH_RIGHT)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.5, color=GRID_C, alpha=0.45)
    ax.set_axisbelow(True)

# ==================== Map ====================
ax_map.set_facecolor("white")

buildings_filtered_km = buildings_filtered.copy()
buildings_filtered_km["geometry"] = buildings_filtered_km.geometry.scale(
    xfact=0.001, yfact=0.001, origin=(0, 0)
)

buildings_filtered_km.plot(ax=ax_map, color=BROWN, edgecolor='none', alpha=0.65, zorder=1)
lc = LineCollection(segments, colors=colors_seg, linewidths=0.5, zorder=2, capstyle='round')
ax_map.add_collection(lc)
ax_map.set_xlim(MAIN_XLIM); ax_map.set_ylim(MAIN_YLIM)
ax_map.set_aspect('equal', anchor='S')
for spine in ax_map.spines.values():
    spine.set_visible(True); spine.set_edgecolor(SPINE_C); spine.set_linewidth(SPINE_WIDTH_MAP)
ax_map.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True,
                   labelsize=8, direction='in', width=SPINE_WIDTH_MAP, colors=TEXT_C)
ax_map.set_xlabel('Easting (km)',  fontsize=10, color=TEXT_C)
ax_map.set_ylabel('Northing (km)', fontsize=10, color=TEXT_C)
ax_map.xaxis.set_major_locator(MultipleLocator(10))
ax_map.yaxis.set_major_locator(MultipleLocator(10))

xlim, ylim = ax_map.get_xlim(), ax_map.get_ylim()
xrange, yrange = xlim[1] - xlim[0], ylim[1] - ylim[0]
bar_len = 2
bar_x0 = xlim[0] + xrange * 0.05
bar_y0 = ylim[0] + yrange * 0.04
tick_h = yrange * 0.008
ax_map.plot([bar_x0, bar_x0 + bar_len], [bar_y0, bar_y0], color=SPINE_C, linewidth=1.2, zorder=10)
ax_map.plot([bar_x0, bar_x0], [bar_y0 - tick_h, bar_y0 + tick_h], color=SPINE_C, linewidth=0.8, zorder=10)
ax_map.plot([bar_x0 + bar_len, bar_x0 + bar_len], [bar_y0 - tick_h, bar_y0 + tick_h], color=SPINE_C, linewidth=0.8, zorder=10)
ax_map.text(bar_x0, bar_y0 + yrange * 0.012, '0', ha='center', va='bottom', fontsize=9, color=TEXT_C, zorder=10)
ax_map.text(bar_x0 + bar_len, bar_y0 + yrange * 0.012, '2 km', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color=TEXT_C, zorder=10)

# ---- 用 node 在地图上标注测量点 ----
for mp in mp_labels:
    node = mp_to_node[mp]
    mask = pipes['_start_str_lookup'] == node
    if mask.any():
        row_data = pipes.loc[mask].iloc[0]
        if pd.notna(row_data.get('us_x')) and pd.notna(row_data.get('us_y')):
            mx, my = row_data['us_x'] / 1000, row_data['us_y'] / 1000
            ax_map.plot(mx, my, marker='^', color=ACCENT_RED, markersize=8,
                        markeredgecolor='white', markeredgewidth=0.7, zorder=5)
            xytext = (1, -14) if mp == 'MP6' else (6, 6)
            ax_map.annotate(mp, (mx, my), textcoords="offset points",
                            xytext=xytext, fontsize=10, fontweight='bold',
                            color=ACCENT_RED, zorder=5)

legend_elements = [
    Line2D([0], [0], color=SAGE_DARK, linewidth=1.5, label=label_map[0]),
    Line2D([0], [0], color=TEAL,      linewidth=1.5, label=label_map[1]),
    Line2D([0], [0], color=NAVY,      linewidth=1.5, label=label_map[2]),
    Patch(facecolor=BROWN, alpha=0.65, label='Buildings'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor=ACCENT_RED,
           markersize=8, markeredgecolor='white', markeredgewidth=0.7,
           label='Measurement Points'),
]
leg_map = ax_map.legend(
    handles=legend_elements,
    loc='upper right',
    fontsize=11,
    frameon=False
)
for t in leg_map.get_texts():
    t.set_color(TEXT_C)

# ==================== 6 个箱型图 ====================
def plot_box_only(ax, data, ylabel, unit='', show_legend=False):
    data = pd.to_numeric(data, errors='coerce').dropna().values
    n = len(data)
    if len(data) < 1:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color=SUBTEXT_C)
        full_label = f'{ylabel} [{unit}]' if unit else ylabel
        ax.set_ylabel(full_label, fontsize=fs_label, color=TEXT_C)
        ax.set_xticks([]); ax.set_xlim(-0.5, 0.5)
        return

    data_clean = remove_outliers_iqr(data, k=1.5)
    if len(data_clean) < 1:
        data_clean = data.copy()

    ax.boxplot(
        data_clean, positions=[0], vert=True, widths=0.4,
        patch_artist=True, showmeans=True,
        meanprops=dict(marker='D', markerfacecolor=MEAN_COLOR,
                       markeredgecolor='white', markersize=4.5, markeredgewidth=0.5),
        medianprops=dict(color=MEDIAN_COLOR, linewidth=MEDIAN_WIDTH),
        boxprops=dict(facecolor=BOX_COLOR, edgecolor=BOX_EDGE,
                      linewidth=BOX_LINEWIDTH, alpha=0.85),
        whiskerprops=dict(color=SPINE_C, linewidth=WHISKER_WIDTH),
        capprops=dict(color=SPINE_C, linewidth=CAP_WIDTH),
        flierprops=dict(marker='', linewidth=0),
        zorder=3
    )

    sns.stripplot(
        x=np.zeros_like(data),
        y=data,
        ax=ax,
        color='#517B76',
        size=0.1,
        alpha=0.5,
        jitter=0.15,
        label='Data' if show_legend else None
    )

    if show_legend:
        leg_handles = [
            Line2D([0], [0], color=MEDIAN_COLOR,
                   linewidth=MEDIAN_WIDTH, label='Median'),
            Line2D([0], [0], marker='D', linestyle='None',
                   markerfacecolor=MEAN_COLOR,
                   markeredgecolor='white',
                   markersize=4.5, label='Mean'),
            Line2D([0], [0], marker='o', linestyle='None',
                   color='#517B76', markersize=3,
                   alpha=0.5, label=f'Data\n(n={n})')
        ]

        leg = ax.legend(
            handles=leg_handles,
            fontsize=7,
            loc='upper right',
            bbox_to_anchor=(1.08, 1.05),
            frameon=False
        )
        for t in leg.get_texts():
            t.set_color(TEXT_C)

    full_label = f'{ylabel} [{unit}]' if unit else ylabel
    ax.set_ylabel(full_label, fontsize=fs_label, color=TEXT_C)
    ax.tick_params(axis='y', labelsize=fs_tick, width=SPINE_WIDTH_RIGHT, colors=TEXT_C)
    ax.set_xticks([]); ax.set_xlim(-0.5, 0.5)

ax_av.set_ylim(-10, 110)
ax_vel.set_ylim(-0.5, 3)
ax_flow.set_ylim(-0.01, 0.06)
ax_diam.set_ylim(-200, 1000)
ax_slp.set_ylim(-0.02, 0.1)
ax_full.set_ylim(-0.05, 1)

plot_box_only(ax_av,   data_av,        'A/V',           unit='1/m',  show_legend=False)
plot_box_only(ax_vel,  data_velocity,  'Velocity',      unit='m/s',  show_legend=True)
plot_box_only(ax_flow, data_flowrate,  'Flowrate',      unit='m³/s', show_legend=False)
plot_box_only(ax_diam, data_diameter,  'Pipe diameter', unit='mm',   show_legend=False)
plot_box_only(ax_slp,  data_slope,     'Slope',         unit='-',    show_legend=False)
plot_box_only(ax_full, data_filling,   'Filling ratio', unit='-',    show_legend=False)

# ==================== Flowrate 验证 ====================
palette_flow = {'Predicted flowrate': COLOR_PRED, 'Measured flowrate': COLOR_MEAS}
sns.barplot(
    data=melted_comparison_df, x='Plot_Name', y='Flowrate_m3d',
    hue='Flowrate_Type',
    hue_order=['Predicted flowrate', 'Measured flowrate'],
    palette=palette_flow, order=display_order,
    linewidth=0, edgecolor='none',
    ax=ax_fval, errorbar=None
)

patches_f = ax_fval.patches
n_g_f = len(display_order)
for i, mp_name in enumerate(display_order):
    bar = patches_f[n_g_f + i]
    bx = bar.get_x() + bar.get_width() / 2
    by = bar.get_height()
    err = std_dict.get(mp_name, 0.0)
    if pd.isna(err):
        err = 0.0
    ax_fval.errorbar(bx, by, yerr=err, fmt='none',
                     ecolor=SPINE_C, elinewidth=ERR_LINE_WIDTH,
                     capsize=2.0, capthick=ERR_LINE_WIDTH)

ax_fval.set_xlabel('Measurement Points', fontsize=fs_label, color=TEXT_C)
ax_fval.set_ylabel('Flowrate [million m³/d]', fontsize=fs_label, color=TEXT_C)
ax_fval.tick_params(labelsize=fs_tick, width=SPINE_WIDTH_RIGHT, colors=TEXT_C)
leg_f = ax_fval.legend(
    title='',
    fontsize=fs_tick,
    loc='upper right',
    frameon=False
)
for t in leg_f.get_texts():
    t.set_color(TEXT_C)

max_flow = melted_comparison_df['Flowrate_m3d'].max()
max_s    = max([v for v in std_dict.values() if pd.notna(v)] + [0.0])
ax_fval.set_ylim(0, (max_flow + max_s) * 1.15)

# ==================== TDS 验证 ====================
palette_tds = {'SHS_in': COLOR_PRED, 'SHS_in_measurement_mean': COLOR_MEAS}
sns.barplot(
    data=plot_df_tds, x='display_name', y='concentration',
    hue='concentration_type',
    hue_order=['SHS_in', 'SHS_in_measurement_mean'],
    palette=palette_tds, order=display_order,
    linewidth=0, edgecolor='none',
    ax=ax_tval, errorbar=None
)

patches_g = ax_tval.patches
n_g_g = len(display_order)
for i, dname in enumerate(display_order):
    bar = patches_g[n_g_g + i]
    bx = bar.get_x() + bar.get_width() / 2
    by = bar.get_height()
    err_val = filtered_df.loc[filtered_df['display_name'] == dname,
                              'SHS_in_measurement_error'].values[0]
    if pd.isna(err_val):
        err_val = 0.0
    ax_tval.errorbar(bx, by, yerr=err_val, fmt='none', ecolor=SPINE_C,
                     capsize=2.0, capthick=ERR_LINE_WIDTH, linewidth=ERR_LINE_WIDTH)

handles_g, labels_g = ax_tval.get_legend_handles_labels()
labels_g = ['Predicted TDS' if l == 'SHS_in'
            else 'Measured TDS' if l == 'SHS_in_measurement_mean'
            else l for l in labels_g]
leg_t = ax_tval.legend(
    handles=handles_g,
    labels=labels_g,
    fontsize=fs_tick,
    loc='upper right',
    frameon=False
)
for t in leg_t.get_texts():
    t.set_color(TEXT_C)

ax_tval.set_xlabel('Measurement Points', fontsize=fs_label, color=TEXT_C)
ax_tval.set_ylabel('TDS [gS/m³]', fontsize=fs_label, color=TEXT_C)
ax_tval.tick_params(labelsize=fs_tick, width=SPINE_WIDTH_RIGHT, colors=TEXT_C)
ax_tval.set_ylim(0, 3.5)
# =============================================================================
# 7. 保存
# =============================================================================
plt.savefig(os.path.join(save_dir, 'Fig1_combined.png'),
            dpi=600, bbox_inches='tight', facecolor=BG_FIG, transparent=False)

if not FIG_ARGS.no_show:
    plt.show()
print(f'\nFigure 1 saved as 600 dpi PNG: {save_dir}Fig1_combined.png')
print(f'HK whisker ranges CSV: {whisker_csv_path}')
