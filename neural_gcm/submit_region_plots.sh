#!/bin/bash
# Generate ALL plots for ALL regions (world + every continent) for all 4 runs,
# into figures/<run>/<region>/<variable>/<family>/. Plot stage ONLY -- the truth
# caches already exist (built by the chunked pipeline), so this just re-scans the
# (small) prediction files to compute the per-region area-means and renders.
# One job per (run, variable); each computes every region in a single pass.
#   EVAL_REGIONS=all -> world + the 7 continents (see eval_common.REGIONS).
# Usage:  bash submit_region_plots.sh
set -euo pipefail
cd "$(dirname "$0")"

RUNS=(nextgems2049 era5_1955 era5_2023 era5_2026)
export EVAL_REGIONS=all
export NG_LEVEL_INTERVAL="${NG_LEVEL_INTERVAL:-50}"

for run in "${RUNS[@]}"; do
  # --time override (longer than the chunked-pipeline default): one job now does
  # 8 regions x 3 families x all levels, so give margin while staying on cpu.
  pj=$(sbatch --parsable --time=00:45:00 \
        --export=ALL,EVAL_RUN="$run",EVAL_REGIONS=all --array=0-4 run_plots.sbatch)
  echo "$run : region-plots=$pj"
done
