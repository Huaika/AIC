#!/bin/bash
#SBATCH --job-name=ng2049_drift
# Small CPU-only job: compute year-mean RMSE + bias vs lead time for the
# NextGEMS-2049 rollouts (verified against NextGEMS-2049 on the model grid) and
# render the twin-axis drift plot. Loads the checkpoint from GCS (compute nodes
# have outbound internet) only to build the regridder. dev_cpu = 30-min fast lane.
#SBATCH --partition=dev_cpu,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --output=results_nextgems_2049/logs/drift_%j.log
#SBATCH --error=results_nextgems_2049/logs/drift_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results_nextgems_2049/logs

PY="$(pwd)/.venv/bin/python"
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "$(date '+%F %T') start on $(hostname) | job ${SLURM_JOB_ID}"
exec "$PY" nextgems_2049_drift.py
