#!/bin/bash
#SBATCH --job-name=ngcm_eval
# Run the (model/data-agnostic) diagnostics for all three runs in parallel, one
# cpu task per run -- far faster than the login node. Each task builds its
# model-grid truth cache once, then generates spaghetti / drift-stat / drift-map
# plots for every requested level. Heavy read (truth cache) -> long cpu walltime.
#   array 0 -> nextgems2049   (also re-makes its plots WITHOUT the spaghetti zoom)
#   array 1 -> era5_1955
#   array 2 -> era5_2023
#   array 3 -> era5_2026   (partial year, Jan..Jun)
# Level set configurable as before (default every 50 hPa):
#   sbatch run_eval.sbatch
#   NG_LEVELS="850,500,200" sbatch run_eval.sbatch
# Re-runs reuse the truth cache + per-level CSVs, so only plotting repeats.
#SBATCH --array=0-3
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=02:30:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/eval_logs/eval_%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/eval_logs/eval_%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p /pfs/work9/workspace/scratch/ka_hc5935-ai-climate/eval_logs

PY="$(pwd)/.venv/bin/python"
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NG_LEVEL_INTERVAL="${NG_LEVEL_INTERVAL:-50}"

RUNS=(nextgems2049 era5_1955 era5_2023 era5_2026)
export EVAL_RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID} run=${EVAL_RUN} "\
"interval=${NG_LEVEL_INTERVAL} levels=${NG_LEVELS:-<interval>}"

# 1) build the shared model-grid truth cache once (no-op if present)
"$PY" -c "import eval_common as c; c.ensure_native_truth()"
# 2) all three plot families, each sweeping every requested level
"$PY" plot_spaghetti.py
"$PY" plot_drift.py
"$PY" plot_drift_maps.py

echo "$(date '+%F %T') done run=${EVAL_RUN}"
