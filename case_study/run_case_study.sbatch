#!/bin/bash
# Heatwave case study: NeuralGCM rollouts over Europe heatwave episodes,
# mixture definition at the 99th percentile (T850 min & max > p99), 2023 & 2026.
# Renders per-episode spaghetti + RMSE-vs-lead (Lagrangian) and a day-10 drift map
# (Eulerian), using only the grid cells in each episode's heatwave footprint.
#SBATCH --job-name=hw_casestudy
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48gb --time=01:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/casestudy_%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/casestudy_%j.log
set -euo pipefail
REPO=/pfs/data6/home/ka/ka_iti/ka_dm9435/code
cd "$REPO"
PY="$REPO/neural_gcm/.venv/bin/python"

export HW_PCT=0.99 HW_CS_DEF=mixture EVAL_VARS="temperature,geopotential" \
       EVAL_REQUIRE_CACHE=1 JAX_PLATFORMS=cpu

for Y in 2023 2026; do
  echo "=================== CASE STUDY $Y ==================="
  EVAL_RUN=era5_$Y HW_YEAR=$Y "$PY" -m aic.controller.casestudy.plots \
    2>&1 | grep -viE "cuInit|GPU|cuda|Tpu|StreamExecutor|Warning|warn"
done
echo "== figure inventory =="
find "$REPO/outputs/figures/case_study" -type f | sort | sed "s#$REPO/##"
