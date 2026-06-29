#!/bin/bash
#SBATCH --job-name=ng2049_plots
# Build the one-time native-level model-grid NextGEMS truth cache (heavy read,
# ~300 GB streamed at ~340 MB/s -> ~15-20 min) then generate ALL diagnostic
# plots (spaghetti, drift stats, drift maps) for every requested pressure level.
# Heavy read -> the long-walltime `cpu` partition (not the 30-min dev lane).
#
# Level set is configurable (default = every 50 hPa, 50..1000):
#   sbatch run_nextgems_2049_plots.sbatch                      # interval 50
#   NG_LEVEL_INTERVAL=100 sbatch run_nextgems_2049_plots.sbatch
#   NG_LEVELS="850,500,200" sbatch run_nextgems_2049_plots.sbatch
# Re-runs reuse the truth cache + per-level CSVs, so only plotting repeats.
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=01:30:00
#SBATCH --output=results_nextgems_2049/logs/plots_%j.log
#SBATCH --error=results_nextgems_2049/logs/plots_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results_nextgems_2049/logs

PY="$(pwd)/.venv/bin/python"
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# pass through the level configuration if set
export NG_LEVEL_INTERVAL="${NG_LEVEL_INTERVAL:-50}"

echo "$(date '+%F %T') start on $(hostname) | job ${SLURM_JOB_ID} | interval=${NG_LEVEL_INTERVAL} levels=${NG_LEVELS:-<interval>}"

# 1) build the shared truth cache once (no-op if already present)
"$PY" -c "import ng2049_common as c; c.ensure_native_truth()"

# 2) all three plot families, each sweeping every requested level
"$PY" nextgems_2049_spaghetti.py
"$PY" nextgems_2049_drift.py
"$PY" nextgems_2049_drift_maps.py

echo "$(date '+%F %T') done"
