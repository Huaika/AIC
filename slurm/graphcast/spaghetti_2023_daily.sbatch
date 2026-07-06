#!/bin/bash
#SBATCH --job-name=gc-spaghetti-2023
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100_short  # use a or h
#SBATCH --gres=gpu:1
#SBATCH --array=0-364%4

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ka/ka_iti/ka_ou7270/AIC}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_DATE="${START_DATE:-2023-01-01}"
INIT_HOUR="${INIT_HOUR:-06}"
SPAGHETTI_ROOT="${SPAGHETTI_ROOT:-${REPO_ROOT}/graphcast/results/spaghetti}"
METRIC_VARIABLE="${METRIC_VARIABLE:-temperature}"
METRIC_LEVEL="${METRIC_LEVEL:-850}"

INIT_DATE="$(date -d "${START_DATE} +${TASK_ID} days" +%Y-%m-%d)"
INIT_TIME="${INIT_DATE}T${INIT_HOUR}:00"
INIT_LABEL="${INIT_TIME//-/}"
INIT_LABEL="${INIT_LABEL//:/}"
INIT_LABEL="${INIT_LABEL//T/_}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-operational}"
RUN_ID="${RUN_ID_PREFIX}_${INIT_LABEL}_10d"
OUT_DIR="${SPAGHETTI_ROOT}/${RUN_ID}"
METRIC_LABEL="${METRIC_VARIABLE// /_}"
METRIC_LABEL="${METRIC_LABEL//\//_}"
if [[ -n "${METRIC_LEVEL}" ]]; then
  METRIC_LABEL="${METRIC_LABEL}_${METRIC_LEVEL}hPa"
fi
EXPECTED_CSV="${OUT_DIR}/${METRIC_LABEL}_global_mean_evolution.csv"

mkdir -p "${REPO_ROOT}/logs" "${SPAGHETTI_ROOT}" "${OUT_DIR}"

if [[ -s "${EXPECTED_CSV}" && "${FORCE:-0}" != "1" ]]; then
  echo "Skipping completed spaghetti task ${TASK_ID}: ${EXPECTED_CSV}"
  exit 0
fi

export USE_NEXTGEMS=0
export USE_ARCO="${USE_ARCO:-1}"
export INIT_TIME
export RUN_ID
export OUT_DIR
export ROLLOUT_STEPS="${ROLLOUT_STEPS:-40}"
export METRICS_FREQUENCY="${METRICS_FREQUENCY:-daily}"
export REFERENCE="${REFERENCE:-auto}"
export METRIC_VARIABLE
export METRIC_LEVEL
export ARCO_CONSOLIDATED="${ARCO_CONSOLIDATED:-false}"
export ARCO_CHUNKS="${ARCO_CHUNKS:-none}"
export CACHE_DIR="${CACHE_DIR:-/scratch/${USER}/graphcast_cache}"
export ARCO_CASE_CACHE_DIR="${ARCO_CASE_CACHE_DIR:-${SPAGHETTI_ROOT}/arco_cases}"
export PYTHON="${PYTHON:-/home/ka/ka_iti/ka_ou7270/miniconda3/envs/graphcast_env/bin/python}"

cd "${REPO_ROOT}"

echo "Spaghetti root: ${SPAGHETTI_ROOT}"
echo "Task id: ${TASK_ID}"
echo "Initialization time: ${INIT_TIME}"
echo "Run id: ${RUN_ID}"
echo "Output directory: ${OUT_DIR}"

# The operational script reads SLURM_SUBMIT_DIR as its repo root.
export SLURM_SUBMIT_DIR="${REPO_ROOT}"
bash graphcast/inference/operational_t2m_rmse.sbatch
