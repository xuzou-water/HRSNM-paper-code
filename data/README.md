# Input data

This directory contains the input datasets required to reproduce the HRSNM
city simulations and Figures 1–5/S4. Generated simulation outputs, caches and
figures are not included.

## Download

Files larger than 50 MiB are stored with Git LFS. Install Git LFS before
cloning, or download the large objects in an existing clone with:

```bash
git lfs install
git lfs pull
```

The complete checkout is approximately 1.6 GB. Verify every input after
download with:

```bash
sha256sum --check data/SHA256SUMS
python -m scripts.validate_repository --check-data
```

## Layout

```text
data/
├── processed_data/
│   ├── hk_v3/
│   │   ├── hydraulic_nodes.csv
│   │   └── hydraulic_results/
│   │       ├── segment_hydraulics.csv
│   │       ├── nodes_all.csv
│   │       └── hk_building_details.csv
│   ├── toronto_v3/
│   │   ├── hydraulic_nodes.csv
│   │   └── hydraulic_results/
│   │       ├── segment_hydraulics.csv
│   │       ├── nodes_all.csv
│   │       └── toronto_building_details.csv
│   └── la_v3/
│       ├── hydraulic_nodes.csv
│       └── hydraulic_results/
│           ├── segment_hydraulics.csv
│           ├── nodes_all.csv
│           └── la_building_details.csv
├── figure1/
│   ├── HK_buildings.shp (+ companion shapefile files)
│   ├── sewer_pipes_filled.csv
│   ├── total_hrt_per_node.csv
│   └── measurement_TDS_update6.csv
├── figure2/
│   ├── 17WWTP_inflow_data1.xlsx
│   ├── HK_border/
│   ├── HK_maintenance/
│   ├── Toronto_border/
│   └── LA_border/
└── global/
    ├── catchment_summary.csv
    └── ne_10m_admin_0_countries/
        └── ne_10m_admin_0_countries.shp (+ companion files)
```

`global/catchment_summary.csv` is the schema-v1 output used by Figure 5. It
contains the service-population quality ratio and daily-flow `Lb/Qb` fields
expected by the plotting code.

## Data terms

Third-party inputs remain subject to their original licences and data-
governance requirements. The files are provided here to support reproduction
of the reported analysis; users are responsible for complying with the
applicable source terms.
