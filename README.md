# HRSNM sewer-sulphide modelling code

Reproducible modelling, input data and figure-generation code for the HRSNM
sewer-sulphide study in Hong Kong, Toronto and Los Angeles. The repository
covers the city models, a 27-scenario environmental grid, main Figures 1–5 and
Supplementary Figure S4.

The input datasets required by the workflow are included under `data/`.
Simulation outputs, caches, generated figures and manuscript files remain
excluded. See [`data/README.md`](data/README.md) for the data inventory and
integrity-check instructions.

For Chinese documentation, see [`README_CN.md`](README_CN.md).

## Repository structure

```text
hrsnm/      shared model components and the three city models
figures/    scripts for Figures 1–5 and Supplementary Figure S4
scripts/    workflow runners, text exporters and validation utilities
tests/      automated model and input checks
data/       versioned input datasets and their integrity manifest
```

Public filenames describe their purpose and do not contain development version
numbers. The numerical methods and calibrated model settings remain in the
corresponding modules.

## Installation

```bash
conda env create -f environment.yml
conda activate sewer
```

Several large inputs are stored with Git LFS. Install Git LFS before cloning,
or run `git lfs pull` in an existing clone to download them.

The workflow is CPU- and memory-based; a GPU is not required.
Typical installation time on a standard desktop computer: Within 30 min

### Software requirements

The code is implemented in Python 3.10. Runtime dependencies include NumPy,
pandas, NetworkX, tqdm, SciPy, Shapely, scikit-learn, Matplotlib,
GeoPandas, openpyxl, seaborn, pyproj and Pillow.

Dependencies and their versions are specified in `environment.yml`.
The commands below use Bash. Git, Git LFS and Conda are required for
the documented installation procedure.

## Reproduce Figures 1–5

After cloning the repository and downloading the LFS objects, run:

```bash
python -m scripts.validate_repository --check-data
./scripts/run_all.sh
```

The full workflow evaluates 81 city-scenario combinations:

```text
3 cities × 27 scenarios
Temperature: 15, 20 and 25 °C
Sulphate:     5, 15 and 25 mg L−1
COD:          250, 525 and 800 mg L−1
```

The total running time in a 1024G memory server is 6 hours.

Outputs are written by default to a sibling directory named
`<repository>_results`. Resource use and output location can be adjusted:

```bash
RESULTS_ROOT=/path/to/results \
SCENARIO_WORKERS=12 \
FIG4_WORKERS=12 \
THREADS_PER_WORKER=1 \
./scripts/run_all.sh
```

Individual modules can also be run from the repository root, for example:

```bash
python -m hrsnm.hong_kong --help
python -m figures.figure_3 --help
python -m scripts.export_figure_1_results --help
```

## Figure 4 and Figure 5 units

- Building-to-outfall distance, `Lb`: m
- Building wastewater flow, `Qb`: m³ d−1
- Conveyance burden, `Σ(Lb/Qb)`: m d m−3 (equivalent to d m−2)

Figure 4 converts a legacy cache only when its metadata explicitly identifies
annual flow units. Figure 5 rejects regression files whose `x_unit` is not
`m day m-3`.

## Validation and tests

Code checks can be run independently of the datasets:

```bash
python -m unittest discover -s tests -v
python -m scripts.validate_repository --require-clean
```

Validate the included data and their expected model structure with:

```bash
python -m scripts.validate_repository --check-data
```

## Data and licence

Third-party datasets remain subject to their original licences and data-
governance requirements. The data are included to support reproduction of the
reported analysis; users remain responsible for complying with the applicable
source terms.

