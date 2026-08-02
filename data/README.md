# External input data

Input data are not included in this repository. Place authorised copies under
`data/` using the following layout before running the full workflow:

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
        └── ne_10m_admin_0_countries.shp (+ companion shapefile files)
```

Run `python -m scripts.validate_repository --check-data` after staging the inputs.
Do not commit these files unless their licences and data-governance conditions
explicitly permit redistribution.
