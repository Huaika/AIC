#!/bin/bash
# Out-of-distribution analysis: NeuralGCM vs GraphCast across the historical (1955),
# present-day (2023) and NextGEMS-2049 future climates. Produces, under
# outputs/figures/out_of_distribution/:
#   * per-year RMSE and bias vs lead (both models overlaid)
#   * one combined multi-year spaghetti (all years + both models + per-era truth)
#SBATCH --job-name=ood_analysis
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=64gb --time=01:15:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/ood_%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/ood_%j.log
set -euo pipefail
REPO=/pfs/data6/home/ka/ka_iti/ka_dm9435/code
cd "$REPO"
PY="$REPO/neural_gcm/.venv/bin/python"

export EVAL_RUNS="era5_1955,era5_2023,nextgems2049,graphcast_era5_1955,graphcast_era5_2023,graphcast_nextgems_2049"
export EVAL_VARS="temperature" NG_LEVELS=850 EVAL_REGIONS=world \
       EVAL_REQUIRE_CACHE=1 JAX_PLATFORMS=cpu

"$PY" -m aic.view.ood 2>&1 | grep -viE "cuInit|GPU|cuda|Tpu|StreamExecutor|Warning|warn"
echo "== outputs =="
find "$REPO/outputs/figures/out_of_distribution" -type f | sort | sed "s#$REPO/##"
