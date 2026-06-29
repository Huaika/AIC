#!/bin/bash
#SBATCH --job-name=ngcm_ng2049
# One array task = one init-day = one 10-day/6h NeuralGCM rollout forced by
# NextGEMS-2049. Many tiny jobs + short walltime + the short partition is
# backfill-optimal: Slurm slots each task into whichever H100 frees first.
#
# Sizing: 10-day rollouts at 1-day stride over 2049 => 355 init-days (idx 0..354).
# No %N cap: let Slurm backfill as many tasks at once as the cluster allows.
#SBATCH --array=0-354
#SBATCH --partition=gpu_h100_short,gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=00:25:00
#SBATCH --output=results_nextgems_2049/logs/task_%A_%a.log
#SBATCH --error=results_nextgems_2049/logs/task_%A_%a.log
#
# Each task reads only the NextGEMS slices it needs (lazy xarray), runs the
# rollout, and writes ONE prediction NetCDF. Resumable: a task whose output
# already exists exits immediately, so re-submitting the same array fills only
# the gaps. GPU-only: JAX_PLATFORMS=cuda makes JAX fail loudly (-> task retried)
# rather than silently fall back to CPU if no GPU is visible.
#
# Submit:        sbatch run_nextgems_2049.sbatch
# Subset/resume: sbatch --array=0-9 run_nextgems_2049.sbatch
#                sbatch --array=0-354 run_nextgems_2049.sbatch      (re-fill gaps)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results_nextgems_2049/logs

PY="$(pwd)/.venv/bin/python"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=default

# This array task processes init-day number SLURM_ARRAY_TASK_ID.
export NG_INIT_INDEX="${SLURM_ARRAY_TASK_ID}"

echo "$(date '+%F %T') start on $(hostname) | job ${SLURM_JOB_ID} | "\
"array ${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID} | "\
"CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

exec "$PY" nextgems_2049_rollout.py
