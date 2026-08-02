# HRSNM 污水管网硫化物模型与绘图代码

本仓库收录 HRSNM 污水管网硫化物研究的可复现代码，涵盖香港、多伦多和洛杉矶
三座城市的模型、27 个环境情景、主文 Figure 1–5，以及补充 Figure S4。

仓库仅发布代码，不包含输入数据、模拟结果、缓存、图片或稿件。外部数据的目录
要求见 [`data/README.md`](data/README.md)。英文说明见
[`README.md`](README.md)。

## 仓库结构

```text
hrsnm/      公共模型模块及三座城市模型
figures/    Figure 1–5 和补充 Figure S4 的绘图脚本
scripts/    批量运行、结果文本导出及仓库检查工具
tests/      模型与输入数据的自动化测试
data/       外部输入数据的放置说明
```

公开文件名已按功能统一，不再保留开发阶段的版本号和 `test` 后缀；模型计算方法
与校准参数仍保留在对应模块中。

## 安装环境

```bash
conda env create -f environment.yml
conda activate sewer
```

工作流使用 CPU 和内存，不需要 GPU。

## 复现 Figure 1–5

按说明将获准使用的数据放入 `data/` 后运行：

```bash
python -m scripts.validate_repository --check-data
./scripts/run_all.sh
```

完整工作流包含 81 个“城市 × 情景”模拟：

```text
3 cities × 27 scenarios
Temperature: 15, 20, 25 °C
Sulphate:     5, 15, 25 mg L−1
COD:          250, 525, 800 mg L−1
```

结果默认写入仓库同级的 `<repository>_results` 目录。可调整输出位置和并行数：

```bash
RESULTS_ROOT=/path/to/results \
SCENARIO_WORKERS=12 \
FIG4_WORKERS=12 \
THREADS_PER_WORKER=1 \
./scripts/run_all.sh
```

也可以在仓库根目录单独运行模块，例如：

```bash
python -m hrsnm.hong_kong --help
python -m figures.figure_3 --help
python -m scripts.export_figure_1_results --help
```

## Figure 4 和 Figure 5 的单位

- 建筑物至排放口距离 `Lb`：m
- 建筑物污水流量 `Qb`：m³ d−1
- 输送负荷指标 `Σ(Lb/Qb)`：m d m−3（等价于 d m−2）

Figure 4 只在旧缓存元数据明确标注年流量单位时进行换算；Figure 5 会拒绝
`x_unit` 不是 `m day m-3` 的回归文件。

## 检查与测试

不放置外部数据也可以运行代码检查：

```bash
python -m unittest discover -s tests -v
python -m scripts.validate_repository --require-clean
```

放置完整数据后，再运行：

```bash
python -m scripts.validate_repository --check-data
```

## 数据与许可

第三方数据仍受原许可和数据治理要求约束；除非明确允许再分发，否则不要提交到
本仓库。目前代码尚未指定开源许可证，在公开仓库前应补充合适的 `LICENSE`。
