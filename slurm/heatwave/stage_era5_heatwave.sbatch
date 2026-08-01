#!/bin/bash
#SBATCH --job-name=hw_t850
# Stage ERA5 850 hPa temperature for the 1991-2020 heatwave baseline, one task per
# YEAR (array 0..29 -> 1991..2020). Streams T850 @00 UTC from ARCO-ERA5 (needs
# GCS/internet, as the rest of the pipeline uses on compute nodes) -> one NetCDF/yr.
# Default cadence = 00 UTC only (~56 GB cloud read/year), whole globe (~0.75 GB/yr).
# Resumable: a finished year .nc is skipped, an interrupted one resumes from .part.
#   sbatch --array=0 --export=ALL,HW_LIMIT=8 slurm/heatwave/stage_era5_heatwave.sbatch  # smoke
#   sbatch slurm/heatwave/stage_era5_heatwave.sbatch                                     # full
#SBATCH --array=0-29
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave_logs/hw_%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave_logs/hw_%A_%a.log
set -euo pipefail
cd /pfs/data6/home/ka/ka_iti/ka_dm9435/code
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave_logs
PY="/pfs/data6/home/ka/ka_iti/ka_dm9435/code/neural_gcm/.venv/bin/python"

export HW_YEAR=$(( 1991 + SLURM_ARRAY_TASK_ID ))
export HW_LEVEL="${HW_LEVEL:-850}"
export HW_CADENCE="${HW_CADENCE:-24}"     # 24 = 00 UTC daily; 6 = 6-hourly
export HW_REGION="${HW_REGION:-world}"
export HW_OUT_DIR="${HW_OUT_DIR:-/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave}"

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID} "\
"array=${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID} year=${HW_YEAR} "\
"level=${HW_LEVEL} cadence=${HW_CADENCE}h region=${HW_REGION}"
exec "$PY" -m aic.controller.staging.stage_era5_heatwave
