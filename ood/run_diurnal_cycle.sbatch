#!/bin/bash
# Diurnal cycle OF THE DATASETS (ERA5 1955, ERA5 2023, NextGEMS 2049): the offset of
# each synoptic hour (00/06/12/18 UTC) from that day's own mean, composited over the
# year, plus the absolute mean field on the twin axis. ONE figure with
#   rows    = geopotential@500, temperature@850
#   columns = 1955, 2023, 2049
# Reads only the cached model-grid truth -- no rollouts are involved.
# Output: outputs/figures/diurnal_cycle/diurnal_cycle/
#SBATCH --job-name=diurnal_cycle
#SBATCH --partition=dev_cpu,dev_cpu_il,cpu,cpu_il
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64gb --time=00:25:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/diurnal_%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/diurnal_%j.log
set -euo pipefail
REPO=/pfs/data6/home/ka/ka_iti/ka_dm9435/code
cd "$REPO"
PY="$REPO/neural_gcm/.venv/bin/python"

export EVAL_RUNS="era5_1955,era5_2023,nextgems2049"
export DIURNAL_SPECS="geopotential:500,temperature:850"
export EVAL_REGIONS="${EVAL_REGIONS:-world,europe}" \
       EVAL_REQUIRE_CACHE=1 JAX_PLATFORMS=cpu

"$PY" -m aic.view.diurnal 2>&1 | grep -viE "cuInit|GPU|cuda|Tpu|StreamExecutor|Warning|warn"
echo "== outputs =="
find "$REPO/outputs/figures/diurnal_cycle" -type f | sort | sed "s#$REPO/##"
