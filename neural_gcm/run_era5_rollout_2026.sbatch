#!/bin/bash
#SBATCH --job-name=era5_roll26
# ERA5-driven NeuralGCM 10-day/6h rollouts for 2026 (partial year), H100 short.
# Each array task = a BATCH of 10 init-days (model+regridder loaded once, then a
# loop) -> ~22-25 min, inside the 30-min short queue.
#
# 2026 staged through 2026-06-23 (ERA5 front). The rollout caps init-days so the
# 10-day forecast stays within data: latest init = last_time - 10d = 2026-06-13,
# i.e. init-days 2026-01-01 .. 2026-06-13 = 164 days -> 17 batches (idx 0..16;
# last batch = 4). init_dates_for_year derives this cap from the staged files, so
# the count is self-correcting -- spare batches just print "nothing to do".
# Resumable: an init-day whose prediction .nc exists is skipped.
#
# Submit AFTER staging completes (or with --dependency=afterok:<stage_jobid>):
#   sbatch --dependency=afterok:<stage_jobid> run_era5_rollout_2026.sbatch
#SBATCH --array=0-16
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_rollout_logs/roll_%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_rollout_logs/roll_%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_rollout_logs

PY="$(pwd)/.venv/bin/python"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=default

export ERA5_YEAR=2026
export ERA5_IN_DIR=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_2026/inputs
export ERA5_OUT_DIR=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_2026/predictions
# batch size overridable for smoke tests: sbatch --export=ALL,ERA5_INIT_COUNT=2 ...
export ERA5_INIT_COUNT=${ERA5_INIT_COUNT:-10}
export ERA5_INIT_START=$(( SLURM_ARRAY_TASK_ID * ERA5_INIT_COUNT ))

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID} "\
"year=${ERA5_YEAR} init_start=${ERA5_INIT_START} count=${ERA5_INIT_COUNT}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

exec "$PY" era5_rollout.py
