#!/usr/bin/env bash
set -euo pipefail

# Self-contained Ubuntu workflow: full27 simulations + Figures 1, 2, 3, 4 and 5.
# Image output is PNG only, at 600 dpi.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-$(command -v python || true)}"
EXPECTED_CONDA_ENV="${CONDA_ENV_NAME:-sewer}"
RESULTS="${RESULTS_ROOT:-${ROOT}_results}"
SCENARIO_WORKERS="${SCENARIO_WORKERS:-81}"
FIG4_WORKERS="${FIG4_WORKERS:-81}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-3}"

if [[ -z "${PYTHON}" ]]; then
  printf 'Python not found. Activate conda environment %s first.\n' \
    "${EXPECTED_CONDA_ENV}" >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "${EXPECTED_CONDA_ENV}" ]]; then
  printf 'Warning: expected conda environment %s; current=%s\n' \
    "${EXPECTED_CONDA_ENV}" "${CONDA_DEFAULT_ENV:-not activated}" >&2
fi

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Prevent 81 scenario processes from each starting a large BLAS thread pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS_PER_WORKER}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREADS_PER_WORKER}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS_PER_WORKER}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${THREADS_PER_WORKER}}"

DATA_BASE="${ROOT}/data/processed_data"
GLOBAL_DIR="${ROOT}/data/global"
FIG1_DATA="${ROOT}/data/figure1"
FIG2_DATA="${ROOT}/data/figure2"
mkdir -p \
  "${RESULTS}/HK_v3/biochemical_results" \
  "${RESULTS}/toronto_v3/biochemical_results" \
  "${RESULTS}/LA_v3/biochemical_results" \
  "${RESULTS}/figure1" "${RESULTS}/figure2" "${RESULTS}/figure3" \
  "${RESULTS}/figure4" "${RESULTS}/figure5" "${RESULTS}/logs"

export HRSNM_HK_SEGMENTS_CSV="${DATA_BASE}/hk_v3/hydraulic_results/segment_hydraulics.csv"
export HRSNM_HK_NODES_CSV="${DATA_BASE}/hk_v3/hydraulic_results/nodes_all.csv"
export HRSNM_TORONTO_SEGMENTS_CSV="${DATA_BASE}/toronto_v3/hydraulic_results/segment_hydraulics.csv"
export HRSNM_TORONTO_NODES_CSV="${DATA_BASE}/toronto_v3/hydraulic_results/nodes_all.csv"
export HRSNM_LA_SEGMENTS_CSV="${DATA_BASE}/la_v3/hydraulic_results/segment_hydraulics.csv"
export HRSNM_LA_NODES_CSV="${DATA_BASE}/la_v3/hydraulic_results/nodes_all.csv"
export HRSNM_FIG2_HK_BORDER_SHP="${FIG2_DATA}/HK_border/hk_merged_border.shp"
export HRSNM_FIG2_HK_MAINT_SHP="${FIG2_DATA}/HK_maintenance/Maintenance_project.shp"
export HRSNM_FIG2_TORONTO_BORDER_SHP="${FIG2_DATA}/Toronto_border/citygcs_regional_mun_wgs84.shp"
export HRSNM_FIG2_LA_BORDER_SHP="${FIG2_DATA}/LA_border/City_Boundary.shp"
export HRSNM_FIG2_MEASURED_XLSX="${FIG2_DATA}/17WWTP_inflow_data1.xlsx"

printf '\n[1/8] Validating package and Python dependencies\n'
"${PYTHON}" -u -m scripts.validate_repository --check-data
"${PYTHON}" -c \
  "import numpy,pandas,networkx,scipy,shapely,sklearn,matplotlib,geopandas,openpyxl,seaborn,pyproj,PIL; print('Python dependency check: PASS')"
"${PYTHON}" -m unittest discover -s tests -q

baseline_complete() {
  local directory="$1" prefix="$2"
  [[ -s "${directory}/${prefix}_result_segments_v7.csv" \
     && -s "${directory}/${prefix}_emission_radius_segments_v7.csv" \
     && -s "${directory}/${prefix}_emission_radius_odour_segments_v7.csv" ]]
}

printf '\n[2/8] Running three baselines concurrently\n'
declare -a baseline_pids=()
declare -a baseline_names=()
declare -a baseline_logs=()

launch_baseline() {
  local city="$1" module="$2" out_dir="$3" prefix="$4"
  local log="${RESULTS}/logs/baseline_${city}.log"
  if baseline_complete "${out_dir}" "${prefix}"; then
    printf '[baseline] %-8s already complete; skipping\n' "${city}"
    return
  fi
  "${PYTHON}" -u -m "${module}" --out-dir "${out_dir}" --skip-shapefile \
    >"${log}" 2>&1 &
  baseline_pids+=("$!")
  baseline_names+=("${city}")
  baseline_logs+=("${log}")
  printf '[baseline] %-8s started as PID %s\n' "${city}" "$!"
}

launch_baseline hk "hrsnm.hong_kong" \
  "${RESULTS}/HK_v3/biochemical_results" hk
launch_baseline toronto "hrsnm.toronto" \
  "${RESULTS}/toronto_v3/biochemical_results" toronto
launch_baseline la "hrsnm.los_angeles" \
  "${RESULTS}/LA_v3/biochemical_results" la

if (( ${#baseline_pids[@]} > 0 )); then
  while true; do
    running=0
    for index in "${!baseline_pids[@]}"; do
      pid="${baseline_pids[$index]}"
      if kill -0 "${pid}" 2>/dev/null; then
        running=$((running + 1))
        printf '[baseline] %-8s PID=%s still running | ' \
          "${baseline_names[$index]}" "${pid}"
        tail -n 1 "${baseline_logs[$index]}" 2>/dev/null || true
      fi
    done
    (( running == 0 )) && break
    sleep 30
  done
  baseline_failed=0
  for index in "${!baseline_pids[@]}"; do
    if ! wait "${baseline_pids[$index]}"; then
      printf '[baseline] %s FAILED. Last log lines:\n' "${baseline_names[$index]}" >&2
      tail -n 30 "${baseline_logs[$index]}" >&2
      baseline_failed=1
    else
      printf '[baseline] %s complete\n' "${baseline_names[$index]}"
    fi
  done
  (( baseline_failed == 0 )) || exit 1
fi

printf '\n[3/8] Running 81 city-scenario jobs (requested workers=%s)\n' \
  "${SCENARIO_WORKERS}"
"${PYTHON}" -u -m scripts.run_parallel_scenarios \
  --root "${ROOT}" --results-root "${RESULTS}" --workers "${SCENARIO_WORKERS}" \
  2>&1 | tee "${RESULTS}/logs/scenario_progress.log"

printf '\n[4/8] Drawing Figure 1 (600 dpi PNG only)\n'
"${PYTHON}" -u -m figures.figure_1 \
  --result-csv "${RESULTS}/HK_v3/biochemical_results/hk_result_segments_v7.csv" \
  --output-dir "${RESULTS}/figure1" \
  --building-shp "${FIG1_DATA}/HK_buildings.shp" \
  --sewer-csv "${FIG1_DATA}/sewer_pipes_filled.csv" \
  --total-hrt-csv "${FIG1_DATA}/total_hrt_per_node.csv" \
  --measurement-csv "${FIG1_DATA}/measurement_TDS_update6.csv" \
  --no-show 2>&1 | tee "${RESULTS}/logs/figure1.log"

printf '\n[5/8] Drawing Figure 2 (600 dpi PNG only)\n'
"${PYTHON}" -u -m figures.figure_2 \
  --results-root "${RESULTS}" \
  --whisker-csv "${RESULTS}/figure1/hk_whisker_ranges.csv" \
  --output-dir "${RESULTS}/figure2" \
  --no-show 2>&1 | tee "${RESULTS}/logs/figure2.log"

printf '\n[6/8] Drawing Figure 3 (600 dpi PNG only)\n'
"${PYTHON}" -u -m figures.figure_3 \
  --results-root "${RESULTS}" --output-dir "${RESULTS}/figure3" --no-show \
  2>&1 | tee "${RESULTS}/logs/figure3.log"

printf '\n[7/8] Drawing Figure 4 with dcorr_dt > 1 mm/year\n'
export HRSNM_FIG4_DATA_BASE="${DATA_BASE}"
export HRSNM_FIG4_HK_BIOCHEM_DIR="${RESULTS}/HK_v3/biochemical_results"
export HRSNM_FIG4_TORONTO_BIOCHEM_DIR="${RESULTS}/toronto_v3/biochemical_results"
export HRSNM_FIG4_LA_BIOCHEM_DIR="${RESULTS}/LA_v3/biochemical_results"
export HRSNM_FIG4_OUT_DIR="${RESULTS}/figure4"
export HRSNM_FIG4_NO_SHOW=1
export HRSNM_FIG4_SCENARIO_GRID=full27
export HRSNM_FIG4_WORKERS="${FIG4_WORKERS}"
export HRSNM_FIG4_EXPECTED_CATCHMENTS=17
cache_count=$(find "${RESULTS}/figure4" -maxdepth 1 -type f \
  -name 'cached_*full27_corr_gt_1mm_per_year*' | wc -l)
if [[ "${cache_count}" -eq 3 ]]; then
  export HRSNM_FIG4_FROM_CACHE=1
  printf 'Three complete Figure 4 caches found; loading them.\n'
else
  export HRSNM_FIG4_FROM_CACHE=0
  printf 'Preprocessing 81 city-scenario tasks (requested workers=%s).\n' \
    "${FIG4_WORKERS}"
fi
"${PYTHON}" -u -m figures.figure_4 \
  2>&1 | tee "${RESULTS}/logs/figure4.log"

printf '\n[8/8] Drawing Figure 5 from the revised Figure 4 regression\n'
export HRSNM_FIG5_GLOBAL_CSV="${GLOBAL_DIR}/catchment_summary.csv"
export HRSNM_FIG5_REGRESSION_CSV="${RESULTS}/figure4/regressed_equations_T_20_SO4_15_COD_525.csv"
export HRSNM_FIG5_COUNTRIES_SHP="${GLOBAL_DIR}/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
export HRSNM_FIG5_OUT_DIR="${RESULTS}/figure5"
export HRSNM_FIG5_NO_SHOW=1
"${PYTHON}" -u -m figures.figure_5 \
  2>&1 | tee "${RESULTS}/logs/figure5.log"

"${PYTHON}" -u -m scripts.verify_figure_outputs --results-root "${RESULTS}"

printf '\nSUCCESS: Figures 1, 2, 3, 4 and 5 are complete.\n'
printf 'Output root: %s\n' "${RESULTS}"
