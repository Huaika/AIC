#!/bin/bash
#SBATCH --job-name=gc-convert-1955
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/ka/ka_iti/ka_ou7270/AIC}"
PYTHON="${PYTHON:-/home/ka/ka_iti/ka_ou7270/miniconda3/envs/graphcast_env/bin/python}"

RAW_DIR="${RAW_DIR:-${REPO_ROOT}/graphcast/data/graphcast/dataset/pre_industrial}"
INIT_TIME="${INIT_TIME:-1955-01-01T06:00}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-40}"
OUTPUT_DATASET="${OUTPUT_DATASET:-${RAW_DIR}/graphcast_1955_init_19550101_06.nc}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-1}"

mkdir -p "${REPO_ROOT}/logs" "${RAW_DIR}"
cd "${REPO_ROOT}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Raw ERA5 directory: ${RAW_DIR}"
echo "Initialization time: ${INIT_TIME}"
echo "Rollout steps: ${ROLLOUT_STEPS}"
echo "Output dataset: ${OUTPUT_DATASET}"
echo "Python: ${PYTHON}"

"${PYTHON}" graphcast/source/convert_era5_to_graphcast_case.py \
  --raw-dir "${RAW_DIR}" \
  --init-time "${INIT_TIME}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --output "${OUTPUT_DATASET}" \
  --compression-level "${COMPRESSION_LEVEL}" \
  --overwrite \
  --log-level INFO

echo "Done. GraphCast-ready dataset:"
echo "${OUTPUT_DATASET}"
