#!/bin/bash
#SBATCH --job-name=era5_stage26
# Stage ERA5 rollout inputs + forcing for 2026, one task per month, Jan..Jun.
# 2026 is the CURRENT year: ARCO-ERA5 is only filled up to its near-real-time
# front. As of 2026-07-28 the front is 2026-07-22 (07-23+ still NaN), so we stage
# months 1..7 and cap July at 2026-07-22 via ERA5_END_DATE.
#   idx 0..6 -> 2026, months 1..7  (Jul = partial, through 2026-07-22 18Z)
# Finished months are skipped, so re-submitting only stages the new/partial month.
# NOTE: to re-stage June with a new end date, delete era5_6hourly_2026_06.nc
# first (a finished month .nc is otherwise skipped).
# Each task writes era5_6hourly_2026_<MM>.nc into MY workspace. Resumable: a
# finished month is skipped; an interrupted one resumes from its .part file.
#SBATCH --array=0-6
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=04:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_staging_logs/stage_%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_staging_logs/stage_%A_%a.log
set -euo pipefail
cd /pfs/data6/home/ka/ka_iti/ka_dm9435/code
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_staging_logs

PY="/pfs/data6/home/ka/ka_iti/ka_dm9435/code/neural_gcm/.venv/bin/python"
export ERA5_YEAR=2026
export ERA5_MONTH=$(( SLURM_ARRAY_TASK_ID + 1 ))
export ERA5_END_DATE=2026-07-22
export ERA5_OUT_DIR=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_2026/inputs

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID} "\
"array=${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID} year=${ERA5_YEAR} month=${ERA5_MONTH} "\
"end=${ERA5_END_DATE}"

exec "$PY" -m ngcm.stage_era5
