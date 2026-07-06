#!/bin/bash
#SBATCH --job-name=ng2049_dmap
# Small CPU-only job: build the Rackow-Fig.3-style global 10-day temperature
# drift maps (NeuralGCM day-10 climatology, NextGEMS reference, and the drift).
# Reuses the cached model-grid truth; reads only the end-of-forecast frames of
# the 355 rollouts. Light + fast. dev_cpu = 30-min fast lane.
#SBATCH --partition=dev_cpu,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=00:30:00
#SBATCH --output=results_nextgems_2049/logs/driftmap_%j.log
#SBATCH --error=results_nextgems_2049/logs/driftmap_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results_nextgems_2049/logs

PY="$(pwd)/.venv/bin/python"
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "$(date '+%F %T') start on $(hostname) | job ${SLURM_JOB_ID}"
exec "$PY" nextgems_2049_drift_maps.py
