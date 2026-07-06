#!/usr/bin/env bash
#
# Run ONE year of the daily-rollout notebook ONCE, on the GPU the scheduler
# assigned via CUDA_VISIBLE_DEVICES. GPU-ONLY: JAX_PLATFORMS=cuda makes JAX
# fail (rather than silently use the CPU) if no GPU is visible, so the
# scheduler treats it as "interrupted" and requeues it for a free GPU later.
#
# Exit 0  = year fully processed (nbconvert ran to completion; done inits skipped)
# Exit !=0 = interrupted (crash / GPU pre-empted / no GPU) -> scheduler resumes it
#
# Usually launched by run_all_daily.sh, not directly.
# Usage:  CUDA_VISIBLE_DEVICES=<gpu> ./run_year_once.sh <year> [gpu_label]
set -uo pipefail
cd "$(dirname "$0")"

YEAR="${1:?usage: run_year_once.sh <year> [gpu_label]}"
GPU="${2:-${CUDA_VISIBLE_DEVICES:-?}}"
PY="$(pwd)/.venv/bin/python"

echo "$(date '+%F %T') [${YEAR}] start on GPU ${GPU} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"      # enforce GPU-only
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=default          # BFC pooling for the long loop
export RACKOW_YEARS="$YEAR"

# Use the venv's built-in 'python3' kernel (lives in .venv/share/jupyter, so it
# is rebuilt with the venv and survives container rebuilds). jupyter_client
# rewrites its argv[0] 'python' to sys.executable = this venv's python, so no
# separate '--user' kernel registration is needed.
exec "$PY" -m nbconvert --to notebook --execute \
  --output "/tmp/exec_daily_${YEAR}.ipynb" \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=-1 \
  rackow_daily_rollouts.ipynb
