#!/bin/bash
# (A) Re-plot the 2026 late-June/July heat-wave episode (ep03) with NeuralGCM +
#     GraphCast on the same plots (split RMSE/bias, side-by-side drift maps).
# (B) Out-of-distribution multi-year spaghetti: NeuralGCM + GraphCast full-year
#     rollouts for 1955, 2023 and 2049 overlaid on one day-of-year axis.
#SBATCH --job-name=hw_cs_ood
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=64gb --time=02:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/cs_ood_%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/cs_ood_%j.log
set -euo pipefail
REPO=/pfs/data6/home/ka/ka_iti/ka_dm9435/code
cd "$REPO"
PY="$REPO/neural_gcm/.venv/bin/python"
export JAX_PLATFORMS=cpu EVAL_REQUIRE_CACHE=1

echo "=================== (A) 2026 ep03 NeuralGCM + GraphCast ==================="
EVAL_SOURCES="neuralgcm,graphcast" EVAL_YEAR=2026 EVAL_DATASET=era5 \
  HW_PCT=0.99 HW_CS_DEF=mixture EVAL_VARS="temperature,geopotential" CS_ONLY_EP=3 \
  "$PY" -m aic.controller.casestudy.plots 2>&1 | grep -viE "cuInit|GPU|cuda|Tpu|StreamExecutor|Warning|warn"

echo "=================== (B) OOD multi-year spaghetti 1955/2023/2049 ==================="
EVAL_RUNS="era5_1955,era5_2023,nextgems2049,graphcast_era5_1955,graphcast_era5_2023,graphcast_nextgems_2049" \
  EVAL_VARS="temperature" NG_LEVELS=850 EVAL_REGIONS=world \
  "$PY" -m aic.view.spaghetti 2>&1 | grep -viE "cuInit|GPU|cuda|Tpu|StreamExecutor|Warning|warn"

echo "== outputs =="
ls "$REPO"/outputs/figures/case_study/mixture_p99/2026/ep03_2026-06-22_2026-07-24/ | sed 's/^/  ep03: /'
find "$REPO"/outputs/figures -name "*multiyear*spaghetti*" | sed "s#$REPO/#  ood: #"
