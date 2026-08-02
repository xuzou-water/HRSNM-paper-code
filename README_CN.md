# HRSNM 论文代码

本仓库整理了 HRSNM 城市污水管网硫化物研究涉及的代码，包括香港、多伦多和
洛杉矶三座城市模型、27 个环境情景、主文 Figure 1–5，以及补充 Figure S4。

该仓库仅包含代码。大型输入数据、受许可限制的数据、模拟结果、缓存、图片和
稿件均不上传。输入文件的目录要求见 [`data/README.md`](data/README.md)。

## 快速运行

```bash
conda env create -f environment.yml
conda activate sewer
chmod +x run_full27_figures_12345.sh
python validate_server_package.py --require-clean
./run_full27_figures_12345.sh
```

完整工作流包括：

```text
3 cities × 27 scenarios = 81 city-scenario simulations
Temperature = 15, 20, 25 °C
SO4         = 5, 15, 25 mg L−1
COD         = 250, 525, 800 mg L−1
```

结果默认写入同级的 `server_full27_package_5_results`。可通过环境变量调整：

```bash
RESULTS_ROOT=/path/to/results \
SCENARIO_WORKERS=12 \
FIG4_WORKERS=12 \
THREADS_PER_WORKER=1 \
./run_full27_figures_12345.sh
```

## 代码组成

- 三城市模型：`hk_HRSNM_v7_5_test2.py`、`toronto_HRSNM_v7_5_test1.py`、
  `la_HRSNM_v7_5_test1.py`
- 公共模型模块：`hrsnm_dissolved_oxygen.py`、`hrsnm_node_mixing.py`、
  `hrsnm_scenarios.py`、`corrosion_criterion.py`
- 主文绘图：Figure 1–5 对应的五个 `HRSNM(...).py` 文件
- 补充图：`plot_figure_s4_50year_failure.py`
- 批量运行：`run_full27_figures_12345.sh`、`run_parallel_scenarios.py`、
  `run_single_scenario.py`
- 检查和测试：`validate_server_package.py`、`verify_figure_outputs.py`、
  `test_*.py`

## Figure 4/5 单位

- `Lb`：m
- `Qb` 及体积流量：m³ d−1
- `Σ(Lb/Qb)`：m d m−3（等价于 d m−2）

Figure 4 会根据明确的 `Q_UNIT` 元数据把旧年度缓存转换为日单位；Figure 5 会
拒绝使用 `x_unit` 不是 `m day m-3` 的旧回归文件。

## 测试

不安装数据时可运行代码检查：

```bash
python -m unittest -q \
  test_hrsnm_dissolved_oxygen.py \
  test_hrsnm_node_mixing.py \
  test_hrsnm_scenarios.py \
  test_corrosion_criterion.py

python validate_server_package.py --require-clean
```

放置完整输入数据后运行：

```bash
python validate_server_package.py --check-data
```

## 许可说明

当前整理包尚未指定开源许可证。公开仓库前应补充合适的 `LICENSE`。第三方数据
仍受各自许可约束，未经允许不应上传到本仓库。
