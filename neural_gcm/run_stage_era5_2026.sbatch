#!/bin/bash
#SBATCH --job-name=era5_stage26
# Stage ERA5 rollout inputs + forcing for 2026, one task per month, Jan..Jun.
# 2026 is the CURRENT year: ARCO-ERA5 is only filled up to its near-real-time
# front (verified 2026-06-23 is the last day with real data; 06-24 onward NaN),
# so we stage only months 1..6 and cap June at 2026-06-23 via ERA5_END_DATE.
#   idx 0..5 -> 2026, months 1..6  (Jun = partial, through 2026-06-23 18Z)
# Each task writes era5_6hourly_2026_<MM>.nc into MY workspace. Resumable: a
# finished month is skipped; an interrupted one resumes from its .part file.
#SBATCH --array=0-5
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=04:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_staging_logs/stage_%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_staging_logs/stage_%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_staging_logs

PY="$(pwd)/.venv/bin/python"
export ERA5_YEAR=2026
export ERA5_MONTH=$(( SLURM_ARRAY_TASK_ID + 1 ))
export ERA5_END_DATE=2026-06-23
export ERA5_OUT_DIR=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_2026/inputs

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID} "\
"array=${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID} year=${ERA5_YEAR} month=${ERA5_MONTH} "\
"end=${ERA5_END_DATE}"

exec "$PY" stage_era5.py
