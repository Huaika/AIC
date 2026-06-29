#!/bin/bash
#SBATCH --job-name=ng2049_spag
# Small CPU-only job: extract global-mean T diagnostics from the finished
# rollouts + the NextGEMS-2049 reference, and render the spaghetti plots.
# dev_cpu is the 30-min fast backfill lane; cpu as fallback. Purely I/O + numpy,
# no GPU. The heavy step is streaming T@1000/850 hPa over the full year from the
# 2.3 TB 3D file (dask-chunked, so memory stays small).
#SBATCH --partition=dev_cpu,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --output=results_nextgems_2049/logs/spaghetti_%j.log
#SBATCH --error=results_nextgems_2049/logs/spaghetti_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results_nextgems_2049/logs

PY="$(pwd)/.venv/bin/python"
export JAX_PLATFORMS=cpu          # no GPU; keep JAX off the accelerator path
export MPLBACKEND=Agg             # headless plotting
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "$(date '+%F %T') start on $(hostname) | job ${SLURM_JOB_ID}"
exec "$PY" nextgems_2049_spaghetti.py
