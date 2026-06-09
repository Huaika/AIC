#!/bin/bash
#SBATCH --job-name=gc-op-1955-10d
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/ka/ka_iti/ka_ou7270/AIC}"
PYTHON="${PYTHON:-/home/ka/ka_iti/ka_ou7270/miniconda3/envs/graphcast_env/bin/python}"

INIT_TIME="${INIT_TIME:-1955-01-01T06:00}"
INIT_LABEL="${INIT_TIME//-/}"
INIT_LABEL="${INIT_LABEL//:/}"
INIT_LABEL="${INIT_LABEL//T/_}"
USE_ARCO="${USE_ARCO:-1}"
LOCAL_DATASET_PATH="${GRAPHCAST_REFERENCE_ERA5:-${REPO_ROOT}/graphcast/data/graphcast/dataset/pre_industrial/graphcast_1955_init_19550101_06.nc}"
RUN_ID="${RUN_ID:-operational_${INIT_LABEL}_10d_${SLURM_JOB_ID:-manual}}"
CACHE_DIR="${CACHE_DIR:-/scratch/${USER}/graphcast_cache}"
ARCO_CASE_CACHE_DIR="${ARCO_CASE_CACHE_DIR:-${CACHE_DIR}/arco_cases}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/graphcast/results/${RUN_ID}}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-40}"
METRICS_FREQUENCY="${METRICS_FREQUENCY:-daily}"
REFERENCE="${REFERENCE:-auto}"
ARCO_CONSOLIDATED="${ARCO_CONSOLIDATED:-false}"
ARCO_CHUNKS="${ARCO_CHUNKS:-none}"
METRIC_VARIABLE="${METRIC_VARIABLE:-2m_temperature}"
METRIC_LEVEL="${METRIC_LEVEL:-}"
METRIC_LABEL="${METRIC_VARIABLE// /_}"
METRIC_LABEL="${METRIC_LABEL//\//_}"
if [[ -n "${METRIC_LEVEL}" ]]; then
  METRIC_LABEL="${METRIC_LABEL}_${METRIC_LEVEL}hPa"
fi

mkdir -p "${REPO_ROOT}/logs" "${CACHE_DIR}" "${OUT_DIR}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export GRAPHCAST_CACHE_DIR="${CACHE_DIR}"
export GRAPHCAST_ROLLOUT_STEPS="${ROLLOUT_STEPS}"
export GRAPHCAST_STEP_HOURS=6
export GRAPHCAST_METRICS_FREQUENCY="${METRICS_FREQUENCY}"
export GRAPHCAST_REFERENCE_MODE="${REFERENCE}"
export GRAPHCAST_METRIC_VARIABLE="${METRIC_VARIABLE}"
if [[ -n "${METRIC_LEVEL}" ]]; then
  export GRAPHCAST_METRIC_LEVEL="${METRIC_LEVEL}"
else
  unset GRAPHCAST_METRIC_LEVEL
fi
export GRAPHCAST_OUTPUT_CSV="${OUT_DIR}/${METRIC_LABEL}_rmse_bias_operational_daily.csv"
export GRAPHCAST_PLOT_PATH="${OUT_DIR}/${METRIC_LABEL}_rmse_bias_operational_daily.png"
export GRAPHCAST_USE_ARCO="${USE_ARCO}"
if [[ "${USE_ARCO,,}" == "1" || "${USE_ARCO,,}" == "true" || "${USE_ARCO,,}" == "yes" ]]; then
  export GRAPHCAST_INIT_TIME="${INIT_TIME}"
  export GRAPHCAST_ARCO_CASE_CACHE_DIR="${ARCO_CASE_CACHE_DIR}"
  export GRAPHCAST_ARCO_ERA5_PATH="${GRAPHCAST_ARCO_ERA5_PATH:-gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3}"
  export GRAPHCAST_ARCO_CONSOLIDATED="${ARCO_CONSOLIDATED}"
  export GRAPHCAST_ARCO_CHUNKS="${ARCO_CHUNKS}"
else
  export GRAPHCAST_REFERENCE_ERA5="${LOCAL_DATASET_PATH}"
fi

cd "${REPO_ROOT}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Use ARCO: ${USE_ARCO}"
echo "Initialization time: ${INIT_TIME}"
if [[ "${USE_ARCO,,}" == "1" || "${USE_ARCO,,}" == "true" || "${USE_ARCO,,}" == "yes" ]]; then
  echo "ARCO source: ${GRAPHCAST_ARCO_ERA5_PATH}"
  echo "ARCO case cache: ${ARCO_CASE_CACHE_DIR}"
  echo "ARCO consolidated metadata: ${ARCO_CONSOLIDATED}"
  echo "ARCO chunks: ${ARCO_CHUNKS}"
else
  echo "Dataset: ${LOCAL_DATASET_PATH}"
fi
echo "Output: ${OUT_DIR}"
echo "Cache: ${CACHE_DIR}"
echo "Rollout steps: ${ROLLOUT_STEPS}"
echo "Metrics frequency: ${METRICS_FREQUENCY}"
echo "Metric variable: ${METRIC_VARIABLE}"
if [[ -n "${METRIC_LEVEL}" ]]; then
  echo "Metric level: ${METRIC_LEVEL} hPa"
fi
echo "Reference: ${REFERENCE}"
echo "Python: ${PYTHON}"
nvidia-smi || true

"${PYTHON}" graphcast/inference/inference_graphcast.py

echo "Done. Results:"
echo "${GRAPHCAST_OUTPUT_CSV}"
echo "${GRAPHCAST_PLOT_PATH}"
