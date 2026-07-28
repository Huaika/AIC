#!/bin/bash
#SBATCH --job-name=gc-rollout-batch
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
# H100-only: the A100-short cards are 40GB and OOM on the 0.25deg/40-step fused
# rollout (needs >40GB GPU; verified by an A100 probe -> RESOURCE_EXHAUSTED).
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
# Two-stage default (GRAPHCAST_RAW_MODE=1): GPU writes UNCOMPRESSED (raw write
# ~14s vs ~270s compressed) -> ~215s/day, so 8 days/task fit the 30-min queue.
# 355 / 8 = 45 tasks. A separate CPU stage (compress_batch.sbatch) compresses the
# raw files. For the old single-stage compressed mode: RAW_MODE=0 + --array=0-118.
#SBATCH --array=0-44
#
# Batched GraphCast daily rollouts -- the NeuralGCM-style driver.
# One array task = ONE process that loads the checkpoint + builds the jitted
# (fused, on-device) predictor ONCE, then loops over a BATCH of init-days.
# This is the analogue of neural_gcm/run_era5_rollout.sbatch.
#
# Quick timing smoke test (2 init-days, 10-day/6h = 40 steps):
#   sbatch --export=ALL,GRAPHCAST_INIT_COUNT=2 graphcast/inference/rollout_batch.sbatch
#
# Full year sweep, batches of ~10 days per task (365 days -> 37 batches):
#   sbatch --array=0-36 graphcast/inference/rollout_batch.sbatch
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# Working env: the neuralgcm jax(cuda12) venv + graphcast deps installed to gc_extra
# (the team graphcast_env at ka_ou7270 is not readable by us). Override on submit
# with --export=ALL,PYTHON=/path/to/other/python if you have your own env.
PYTHON="${PYTHON:-$REPO_ROOT/neural_gcm/.venv/bin/python}"
# Set unconditionally (NOT ${PYTHONPATH:-...}): the gate submits with --export=ALL,
# which propagates its own PYTHONPATH=graphcast/inference and would otherwise shadow
# gc_extra + graphcast/source, breaking `import graphcast`.
export PYTHONPATH="/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_extra:$REPO_ROOT/src/aic/model/graphcast/source:$REPO_ROOT/src/aic/model/graphcast/inference"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

# ---- what to run ------------------------------------------------------------
export GRAPHCAST_YEAR="${GRAPHCAST_YEAR:-2023}"
export GRAPHCAST_INIT_HOUR="${GRAPHCAST_INIT_HOUR:-6}"
export GRAPHCAST_INIT_STRIDE_DAYS="${GRAPHCAST_INIT_STRIDE_DAYS:-1}"
export GRAPHCAST_ROLLOUT_STEPS="${GRAPHCAST_ROLLOUT_STEPS:-40}"   # 40*6h = 10 days
export GRAPHCAST_STEP_HOURS="${GRAPHCAST_STEP_HOURS:-6}"

# Batch selection: this task runs [START:START+COUNT] of the year's init-days.
export GRAPHCAST_RAW_MODE="${GRAPHCAST_RAW_MODE:-1}"
if [[ "$GRAPHCAST_RAW_MODE" == "1" ]]; then
  INIT_COUNT="${GRAPHCAST_INIT_COUNT:-8}"      # raw write is cheap -> more days/job
else
  INIT_COUNT="${GRAPHCAST_INIT_COUNT:-3}"
fi
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
export GRAPHCAST_INIT_COUNT="${INIT_COUNT}"
export GRAPHCAST_INIT_START="${GRAPHCAST_INIT_START:-$(( TASK_ID * INIT_COUNT ))}"

# ---- data source: ERA5/ARCO (default) or NextGEMS -----------------------------
export GRAPHCAST_USE_NEXTGEMS="${GRAPHCAST_USE_NEXTGEMS:-0}"
if [[ "${GRAPHCAST_USE_NEXTGEMS}" == "1" ]]; then
  export GRAPHCAST_USE_ARCO=0
  export GRAPHCAST_NEXTGEMS_YEAR="${GRAPHCAST_NEXTGEMS_YEAR:-${GRAPHCAST_YEAR}}"
  export GRAPHCAST_NEXTGEMS_ROOT="${GRAPHCAST_NEXTGEMS_ROOT:-/pfs/work9/workspace/scratch/ka_je2428-nextgems_${GRAPHCAST_NEXTGEMS_YEAR}}"
  export GRAPHCAST_NEXTGEMS_CASE_CACHE_DIR="${GRAPHCAST_NEXTGEMS_CASE_CACHE_DIR:-/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/graphcast_nextgems_cases}"
  export GRAPHCAST_SCENARIO_LABEL="${GRAPHCAST_SCENARIO_LABEL:-nextgems_${GRAPHCAST_NEXTGEMS_YEAR}}"
  DATASET_TAG="nextgems_${GRAPHCAST_NEXTGEMS_YEAR}"
else
  export GRAPHCAST_USE_ARCO="${GRAPHCAST_USE_ARCO:-1}"
  export GRAPHCAST_ARCO_ERA5_PATH="${GRAPHCAST_ARCO_ERA5_PATH:-gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3}"
  export GRAPHCAST_ARCO_CONSOLIDATED="${GRAPHCAST_ARCO_CONSOLIDATED:-false}"
  export GRAPHCAST_ARCO_CHUNKS="${GRAPHCAST_ARCO_CHUNKS:-none}"
  DATASET_TAG="era5_${GRAPHCAST_YEAR}"
fi

# ---- metric + outputs -------------------------------------------------------
export GRAPHCAST_METRIC_VARIABLE="${GRAPHCAST_METRIC_VARIABLE:-temperature}"
export GRAPHCAST_METRIC_LEVEL="${GRAPHCAST_METRIC_LEVEL:-850}"
export GRAPHCAST_METRICS_FREQUENCY="${GRAPHCAST_METRICS_FREQUENCY:-daily}"
export GRAPHCAST_REFERENCE_MODE="${GRAPHCAST_REFERENCE_MODE:-auto}"
export GRAPHCAST_WRITE_NETCDF="${GRAPHCAST_WRITE_NETCDF:-1}"
export GRAPHCAST_RUN_PREFIX="${GRAPHCAST_RUN_PREFIX:-operational}"
# Phase 1: write ONLY the prediction .nc files (metric CSVs + plots are phase 2).
export GRAPHCAST_NO_PLOT="${GRAPHCAST_NO_PLOT:-1}"
export GRAPHCAST_NETCDF_ONLY="${GRAPHCAST_NETCDF_ONLY:-1}"
# NeuralGCM-compatible output: flat pred_<year>_<YYYY-MM-DD>.nc with
# latitude/longitude coords, so the same eval/figure pipeline applies.
export GRAPHCAST_NEURALGCM_LAYOUT="${GRAPHCAST_NEURALGCM_LAYOUT:-1}"

CACHE_DIR="${CACHE_DIR:-/scratch/${USER}/graphcast_cache}"
export GRAPHCAST_CACHE_DIR="${CACHE_DIR}"
export GRAPHCAST_ARCO_CASE_CACHE_DIR="${GRAPHCAST_ARCO_CASE_CACHE_DIR:-/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/graphcast_arco_cases}"
# Predictions written to the WORKSPACE, never the git repo tree. Mirrors
# NeuralGCM's <dataset>_<year>/predictions layout. In two-stage RAW mode the GPU
# writes UNCOMPRESSED into predictions_raw/, and compress_batch.sbatch produces
# the final compressed predictions/ (GRAPHCAST_FINAL_DIR lets the GPU skip a day
# whose compressed final already exists).
GC_BASE="/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/graphcast_${DATASET_TAG}"
if [[ "$GRAPHCAST_RAW_MODE" == "1" ]]; then
  export GRAPHCAST_WRITE_COMPLEVEL="${GRAPHCAST_WRITE_COMPLEVEL:-0}"
  export GRAPHCAST_FINAL_DIR="${GRAPHCAST_FINAL_DIR:-${GC_BASE}/predictions}"
  export GRAPHCAST_OUT_DIR="${GRAPHCAST_OUT_DIR:-${GC_BASE}/predictions_raw}"
else
  export GRAPHCAST_OUT_DIR="${GRAPHCAST_OUT_DIR:-${GC_BASE}/predictions}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_DIR}/matplotlib}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

mkdir -p "${REPO_ROOT}/logs" "${CACHE_DIR}" "${GRAPHCAST_OUT_DIR}"
cd "${REPO_ROOT}"

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-manual} array=${SLURM_ARRAY_TASK_ID:-0}"
echo "year=${GRAPHCAST_YEAR} init_start=${GRAPHCAST_INIT_START} count=${GRAPHCAST_INIT_COUNT}"
echo "rollout_steps=${GRAPHCAST_ROLLOUT_STEPS} step_hours=${GRAPHCAST_STEP_HOURS}"
echo "out=${GRAPHCAST_OUT_DIR}"
echo "python=${PYTHON}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

exec "${PYTHON}" src/aic/model/graphcast/inference/inference_graphcast.py
