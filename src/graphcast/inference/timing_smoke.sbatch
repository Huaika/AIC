#!/bin/bash
#SBATCH --job-name=gc-timing-smoke
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_timing_logs/%x-%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_timing_logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#
# Timing smoke test for the refactored (fused + batched) GraphCast rollout.
# Runs GRAPHCAST_INIT_COUNT init-days in ONE process: day 1 pays the JIT
# compile, day 2 is compile-free -> shows the amortized per-day rollout cost.
set -euo pipefail

REPO_ROOT="/pfs/data6/home/ka/ka_iti/ka_dm9435/code"
cd "$REPO_ROOT"
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_timing_logs

# Reuse the working neuralgcm jax(cuda12) venv + graphcast deps installed to gc_extra.
PY="$REPO_ROOT/neural_gcm/.venv/bin/python"
export PYTHONPATH="/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_extra:$REPO_ROOT/graphcast/source:$REPO_ROOT/graphcast/inference"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_CPP_MIN_LOG_LEVEL=1

CACHE_DIR=/scratch/${USER}/graphcast_cache
export GRAPHCAST_CACHE_DIR="$CACHE_DIR"
# Persistent (NOT /scratch): reuse precache_arco_batch.sbatch cases across jobs.
export GRAPHCAST_ARCO_CASE_CACHE_DIR="${GRAPHCAST_ARCO_CASE_CACHE_DIR:-/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/graphcast_arco_cases}"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
mkdir -p "$CACHE_DIR"

# What to run: 2 init-days, 10-day/6h rollout (40 steps), ARCO ERA5 2023.
export GRAPHCAST_YEAR=2023
export GRAPHCAST_INIT_HOUR=6
export GRAPHCAST_INIT_START="${GRAPHCAST_INIT_START:-0}"
export GRAPHCAST_INIT_COUNT="${GRAPHCAST_INIT_COUNT:-2}"
export GRAPHCAST_ROLLOUT_STEPS="${GRAPHCAST_ROLLOUT_STEPS:-40}"
export GRAPHCAST_STEP_HOURS=6
export GRAPHCAST_USE_ARCO=1
export GRAPHCAST_METRIC_VARIABLE=temperature
export GRAPHCAST_METRIC_LEVEL=850
export GRAPHCAST_WRITE_NETCDF="${GRAPHCAST_WRITE_NETCDF:-1}"
export GRAPHCAST_NO_PLOT="${GRAPHCAST_NO_PLOT:-1}"
export GRAPHCAST_OUT_DIR=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_timing_out

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "count=$GRAPHCAST_INIT_COUNT steps=$GRAPHCAST_ROLLOUT_STEPS out=$GRAPHCAST_OUT_DIR"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

exec "$PY" graphcast/inference/inference_graphcast.py
