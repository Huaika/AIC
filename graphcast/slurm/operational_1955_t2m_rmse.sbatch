#!/bin/bash
#SBATCH --job-name=gc-op-1955-rmse
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/ka/ka_iti/ka_ou7270/AIC}"
PYTHON="${PYTHON:-/home/ka/ka_iti/ka_ou7270/miniconda3/envs/graphcast_env/bin/python}"

# Optional: GraphCast-formatted 1955 NetCDF containing 2 input frames plus
# 40 future truth frames if you want true 10-day ground-truth RMSE. If unset,
# the Python script auto-selects a public DeepMind sample dataset for smoke
# testing only.
DATASET_ARGS=()
DATASET_LABEL="auto-selected DeepMind sample dataset"
if [[ -n "${GRAPHCAST_1955_DATASET:-}" ]]; then
  DATASET_ARGS=(--dataset-path "${GRAPHCAST_1955_DATASET}")
  DATASET_LABEL="${GRAPHCAST_1955_DATASET}"
fi

RUN_ID="${RUN_ID:-operational_1955_t2m_rmse_${SLURM_JOB_ID:-manual}}"
CACHE_DIR="${CACHE_DIR:-/scratch/${USER}/graphcast_cache}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/graphcast/results/${RUN_ID}}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-40}"
METRICS_FREQUENCY="${METRICS_FREQUENCY:-daily}"
if [[ -z "${REFERENCE:-}" ]]; then
  if [[ -n "${GRAPHCAST_1955_DATASET:-}" ]]; then
    REFERENCE="ground_truth"
  else
    REFERENCE="initial_state"
  fi
fi

mkdir -p "${REPO_ROOT}/logs" "${CACHE_DIR}" "${OUT_DIR}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

cd "${REPO_ROOT}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Dataset: ${DATASET_LABEL}"
echo "Output: ${OUT_DIR}"
echo "Cache: ${CACHE_DIR}"
echo "Rollout steps: ${ROLLOUT_STEPS}"
echo "Metrics frequency: ${METRICS_FREQUENCY}"
echo "Reference: ${REFERENCE}"
echo "Python: ${PYTHON}"
nvidia-smi || true

"${PYTHON}" graphcast/source/run_t2m_drift_rollout.py \
  --model operational \
  "${DATASET_ARGS[@]}" \
  --reference "${REFERENCE}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --step-hours 6 \
  --metrics-frequency "${METRICS_FREQUENCY}" \
  --cache-dir "${CACHE_DIR}" \
  --output-csv "${OUT_DIR}/t2m_rmse_bias_operational_1955_daily.csv" \
  --plot-path "${OUT_DIR}/t2m_rmse_bias_operational_1955_daily.png" \
  --xla-mem-fraction "${XLA_PYTHON_CLIENT_MEM_FRACTION}" \
  --log-level INFO

echo "Done. Results:"
echo "${OUT_DIR}/t2m_rmse_bias_operational_1955_daily.csv"
echo "${OUT_DIR}/t2m_rmse_bias_operational_1955_daily.png"
