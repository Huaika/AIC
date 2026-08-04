#!/bin/bash
#SBATCH --job-name=gc-arco-2023-subset
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ka/ka_iti/ka_ou7270/AIC}"
PYTHON="${PYTHON:-/home/ka/ka_iti/ka_ou7270/miniconda3/envs/graphcast_env/bin/python}"
SPAGHETTI_ROOT="${SPAGHETTI_ROOT:-${REPO_ROOT}/graphcast/results/spaghetti}"
LOCAL_ARCO_PATH="${LOCAL_ARCO_PATH:-${SPAGHETTI_ROOT}/arco_2023_6h_required.zarr}"
START_TIME="${START_TIME:-2023-01-01T00:00}"
END_TIME="${END_TIME:-2024-01-10T06:00}"
TIME_CHUNK="${TIME_CHUNK:-4}"

mkdir -p "${REPO_ROOT}/logs" "${SPAGHETTI_ROOT}"

export GRAPHCAST_ARCO_CONSOLIDATED="${GRAPHCAST_ARCO_CONSOLIDATED:-false}"
export GRAPHCAST_ARCO_CHUNKS="${GRAPHCAST_ARCO_CHUNKS:-auto}"

cd "${REPO_ROOT}"

echo "Local ARCO subset: ${LOCAL_ARCO_PATH}"
echo "Time range: ${START_TIME} to ${END_TIME}"
echo "Time chunk: ${TIME_CHUNK}"
echo "Python: ${PYTHON}"

args=(
  graphcast/inference/cache_arco_year_subset.py
  --output "${LOCAL_ARCO_PATH}"
  --start "${START_TIME}"
  --end "${END_TIME}"
  --time-chunk "${TIME_CHUNK}"
)

if [[ "${FORCE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

"${PYTHON}" "${args[@]}"
