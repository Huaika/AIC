#!/bin/bash
#SBATCH --job-name=era5_stage
# Stage the ERA5 rollout inputs + forcing (6-hourly, 7 input vars @ 37 levels +
# sst/ci, native 0.25 deg) into local storage, one task per (year, month).
# 2 years x 12 months = 24 tasks (no concurrency cap; the scheduler runs as many
# as it can backfill). One month (~134 GB, ~21 min at ~108 MB/s) is a good per-job
# size: substantial work, short walltime, backfill-friendly.
#   idx 0..11  -> 1955, months 1..12
#   idx 12..23 -> 2023, months 1..12
# Each task writes era5_6hourly_<year>_<MM>.nc. Resumable: a finished month is
# skipped; an interrupted one resumes from its .part file.
#SBATCH --array=0-23
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=04:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_staging_logs/stage_%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_staging_logs/stage_%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p /pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_staging_logs

PY="$(pwd)/.venv/bin/python"
YEARS=(1955 2023)
i=${SLURM_ARRAY_TASK_ID}
export ERA5_YEAR=${YEARS[$(( i / 12 ))]}
export ERA5_MONTH=$(( i % 12 + 1 ))

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID} "\
"array=${SLURM_ARRAY_JOB_ID}:${i} year=${ERA5_YEAR} month=${ERA5_MONTH}"

exec "$PY" stage_era5.py
