#!/bin/bash
#SBATCH --job-name=neuralgcm_pm2
# Short walltime + multiple partitions so Slurm backfills into whichever H100
# slot frees first (gpu_h100_short is the dedicated 30-min fast lane; gpu_h100
# also accepts a 30-min job). Backfill favours short jobs -> shorter queue wait.
#SBATCH --partition=gpu_h100_short,gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=results/pretrained_model_2_%j.log
#SBATCH --error=results/pretrained_model_2_%j.log
#
# Execute neural_gcm/pretrained_model_2.ipynb headless on one H100.
# The model checkpoint + ERA5 slice stream from public GCS buckets at runtime
# (UC3 compute nodes have outbound internet, verified). GPU-only: JAX_PLATFORMS=cuda
# makes JAX fail loudly rather than silently fall back to CPU if no GPU is visible.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PY="$(pwd)/.venv/bin/python"

echo "$(date '+%F %T') start on $(hostname) | job $SLURM_JOB_ID | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Execute to a separate output notebook so the git-tracked original stays clean.
exec "$PY" -m nbconvert --to notebook --execute \
  --output "pretrained_model_2.executed.ipynb" \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=-1 \
  pretrained_model_2.ipynb
