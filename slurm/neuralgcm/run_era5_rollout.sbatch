#!/bin/bash
#SBATCH --job-name=era5_rollout
# ERA5-driven NeuralGCM rollouts for 1955 + 2023, on short-term H100 instances.
# Each array task = a BATCH of 10 init-days (model+regridder loaded once, then a
# loop) -> ~10 x ~2:15 ~= 22-25 min, comfortably inside the 30-min short queue,
# and far fewer jobs/overhead than one-init-per-task.
#
# 355 init-days/year (Jan-01 .. Dec-21) -> 36 batches/year (idx 0..35; last = 5).
#   array idx  0..35  -> 1955, batch idx = id
#   array idx 36..71  -> 2023, batch idx = id-36
# Resumable: an init-day whose prediction .nc exists is skipped, so re-submitting
# the same array only fills gaps.
#SBATCH --array=0-71
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_rollout_logs/roll_%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_rollout_logs/roll_%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p /pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_rollout_logs

PY="$(pwd)/.venv/bin/python"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=default

i=${SLURM_ARRAY_TASK_ID}
if [ "$i" -lt 36 ]; then export ERA5_YEAR=1955; B=$i; else export ERA5_YEAR=2023; B=$((i - 36)); fi
# batch size overridable for smoke tests: sbatch --export=ALL,ERA5_INIT_COUNT=2 ...
export ERA5_INIT_COUNT=${ERA5_INIT_COUNT:-10}
export ERA5_INIT_START=$(( B * ERA5_INIT_COUNT ))

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}:${i} "\
"year=${ERA5_YEAR} init_start=${ERA5_INIT_START} count=${ERA5_INIT_COUNT}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

exec "$PY" era5_rollout.py
