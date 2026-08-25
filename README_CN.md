# HRSNM 污水管网硫化物模型、数据与绘图代码

本仓库收录 HRSNM 污水管网硫化物研究的可复现代码和输入数据，涵盖香港、多伦多
和洛杉矶三座城市的模型、27 个环境情景、主文 Figure 1–5，以及补充
Figure S4。

工作流所需的输入数据已收录在 `data/` 目录；模拟结果、缓存、生成图片和稿件仍不包含在
仓库内。数据清单和完整性检查方法见 [`data/README.md`](data/README.md)。英文说明见
[`README.md`](README.md)。

## 仓库结构

```text
hrsnm/      公共模型模块及三座城市模型
figures/    Figure 1–5 和补充 Figure S4 的绘图脚本
scripts/    批量运行、结果文本导出及仓库检查工具
tests/      模型与输入数据的自动化测试
data/       版本化的输入数据及完整性校验清单
```

公开文件名已按功能统一，不再保留开发阶段的版本号和 `test` 后缀；模型计算方法
与校准参数仍保留在对应模块中。

## 安装环境

```bash
conda env create -f environment.yml
conda activate sewer
```

部分大文件使用 Git LFS 存储。请在克隆前安装 Git LFS；已有克隆可在仓库目录运行
`git lfs pull` 下载大文件。

工作流使用 CPU 和内存，不需要 GPU。

## 复现 Figure 1–5

克隆仓库并下载 LFS 对象后运行：

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

代码检查可以独立于数据运行：

```bash
python -m unittest discover -s tests -v
python -m scripts.validate_repository --require-clean
```

使用下列命令验证已包含的数据及其模型结构：

```bash
python -m scripts.validate_repository --check-data
```

## 数据与许可

第三方数据仍受原许可和数据治理要求约束。仓库中的数据用于复现论文报告的分析；使用者仍需
遵守各数据源的适用条款。目前代码尚未指定开源许可证，在公开仓库前应补充合适的 `LICENSE`。
