#!/bin/bash
#SBATCH --job-name=gc-arco-cache-2023
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --array=0-364%4

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ka/ka_iti/ka_ou7270/AIC}"
PYTHON="${PYTHON:-/home/ka/ka_iti/ka_ou7270/miniconda3/envs/graphcast_env/bin/python}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_DATE="${START_DATE:-2023-01-01}"
INIT_HOUR="${INIT_HOUR:-06}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-40}"
STEP_HOURS="${STEP_HOURS:-6}"
CACHE_DIR="${CACHE_DIR:-/scratch/${USER}/graphcast_cache}"
SPAGHETTI_ROOT="${SPAGHETTI_ROOT:-${REPO_ROOT}/graphcast/results/spaghetti}"
ARCO_CASE_CACHE_DIR="${ARCO_CASE_CACHE_DIR:-${SPAGHETTI_ROOT}/arco_cases}"
ARCO_CONSOLIDATED="${ARCO_CONSOLIDATED:-false}"
ARCO_CHUNKS="${ARCO_CHUNKS:-none}"
ARCO_COMPRESSION_LEVEL="${ARCO_COMPRESSION_LEVEL:-1}"

INIT_DATE="$(date -d "${START_DATE} +${TASK_ID} days" +%Y-%m-%d)"
INIT_TIME="${INIT_DATE}T${INIT_HOUR}:00"
CASE_LABEL="${INIT_DATE//-/}_${INIT_HOUR}"
EXPECTED_CASE="${ARCO_CASE_CACHE_DIR}/graphcast_arco_${CASE_LABEL}_steps${ROLLOUT_STEPS}_h${STEP_HOURS}.nc"

mkdir -p "${REPO_ROOT}/logs" "${ARCO_CASE_CACHE_DIR}"

export GRAPHCAST_ARCO_CASE_CACHE_DIR="${ARCO_CASE_CACHE_DIR}"
export GRAPHCAST_ARCO_CONSOLIDATED="${ARCO_CONSOLIDATED}"
export GRAPHCAST_ARCO_CHUNKS="${ARCO_CHUNKS}"
export GRAPHCAST_ARCO_COMPRESSION_LEVEL="${ARCO_COMPRESSION_LEVEL}"

cd "${REPO_ROOT}"

echo "Task id: ${TASK_ID}"
echo "Initialization time: ${INIT_TIME}"
echo "Expected case: ${EXPECTED_CASE}"
echo "ARCO case cache: ${ARCO_CASE_CACHE_DIR}"
echo "ARCO consolidated metadata: ${ARCO_CONSOLIDATED}"
echo "ARCO chunks: ${ARCO_CHUNKS}"
echo "Python: ${PYTHON}"

args=(
  graphcast/inference/precache_arco_case.py
  --init-time "${INIT_TIME}"
  --rollout-steps "${ROLLOUT_STEPS}"
  --step-hours "${STEP_HOURS}"
  --case-cache-dir "${ARCO_CASE_CACHE_DIR}"
  --compression-level "${ARCO_COMPRESSION_LEVEL}"
)

if [[ "${FORCE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

"${PYTHON}" "${args[@]}"
