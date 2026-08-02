# HRSNM paper code

This repository contains the modelling, analysis and plotting code used for the
HRSNM sewer-sulphide study. It covers the three city models (Hong Kong,
Toronto and Los Angeles), the 27-scenario workflow, main Figures 1–5 and
Supplementary Figure 4.

The repository is code-only. Large, licensed or location-sensitive input data,
simulation outputs, caches and manuscript files are intentionally excluded.
See [`data/README.md`](data/README.md) for the expected input layout.

## Analysis workflow

The full analysis comprises 81 city-scenario simulations:

```text
3 cities × 27 scenarios
Temperature: 15, 20 and 25 °C
Sulphate:     5, 15 and 25 mg L−1
COD:          250, 525 and 800 mg L−1
```

The principal entry point runs model validation, three baseline simulations,
the scenario grid and Figures 1–5:

```bash
conda env create -f environment.yml
conda activate sewer
chmod +x run_full27_figures_12345.sh
python validate_server_package.py --require-clean
./run_full27_figures_12345.sh
```

Generated outputs are written by default to a sibling directory named
`server_full27_package_5_results`. Override this location and worker counts as
needed:

```bash
RESULTS_ROOT=/path/to/results \
SCENARIO_WORKERS=12 \
FIG4_WORKERS=12 \
THREADS_PER_WORKER=1 \
./run_full27_figures_12345.sh
```

The workflow is CPU- and memory-based and does not require a GPU.

## Code map

| Component | Files |
| --- | --- |
| City models | `hk_HRSNM_v7_5_test2.py`, `toronto_HRSNM_v7_5_test1.py`, `la_HRSNM_v7_5_test1.py` |
| Shared model logic | `hrsnm_dissolved_oxygen.py`, `hrsnm_node_mixing.py`, `hrsnm_scenarios.py`, `corrosion_criterion.py` |
| Main figures | `HRSNM(v7 Fig1_4_test2).py`, `HRSNM(v Fig2_9).py`, `HRSNM(v7 Figure3).py`, `HRSNM(v7 Fig4_2).py`, `HRSNM(v7 Fig5_3).py` |
| Supplementary figure | `plot_figure_s4_50year_failure.py` |
| Orchestration | `run_full27_figures_12345.sh`, `run_parallel_scenarios.py`, `run_single_scenario.py` |
| QA and tests | `validate_server_package.py`, `verify_figure_outputs.py`, `test_*.py` |
| Results-text helpers | `export_fig1_results_text.py`, `export_fig2_results_text.py` |

## Unit convention for the conveyance burden index

The current Figure 4 and Figure 5 implementation uses:

- building-to-outfall distance, `Lb`: m;
- building wastewater flow, `Qb`: m³ d−1;
- `Σ(Lb/Qb)`: m d m−3 (equivalent to d m−2).

Figure 4 can migrate legacy caches explicitly labelled `m3/year` to daily
units. Figure 5 rejects a stale regression CSV unless its `x_unit` is
`m day m-3`.

## Tests

Code-only checks do not require the external dataset:

```bash
python -m unittest -q \
  test_hrsnm_dissolved_oxygen.py \
  test_hrsnm_node_mixing.py \
  test_hrsnm_scenarios.py \
  test_corrosion_criterion.py

python validate_server_package.py --require-clean
```

After the data have been staged, validate the complete input layout with:

```bash
python validate_server_package.py --check-data
```

## Data and licensing

No licence is asserted in this code release yet. Add an appropriate `LICENSE`
file before making the repository public. Third-party datasets remain subject
to their original licences and should not be redistributed through this
repository without permission.
