#!/bin/bash
#SBATCH --job-name=ngcm_eval
# Run the (model/data-agnostic) diagnostics for all four runs in parallel, one
# cpu task per run -- far faster than the login node. Each task builds its
# per-variable model-grid truth cache(s) once (in a single pass over the source),
# then generates spaghetti / drift-stat / drift-map plots for every requested
# (variable, level). Heavy read -> long cpu walltime.
#   array 0 -> nextgems2049
#   array 1 -> era5_1955
#   array 2 -> era5_2023
#   array 3 -> era5_2026   (partial year, Jan..Jun)
# Level set configurable (default every 50 hPa) AND variable set configurable
# (default: the 5 core prognostics T,Z,Q,U,V; see VARIABLES in eval_common.py):
#   sbatch run_eval.sbatch
#   NG_LEVELS="850,500,200" sbatch run_eval.sbatch
#   EVAL_VARS="temperature,geopotential,specific_humidity,u_component_of_wind,v_component_of_wind,specific_cloud_ice_water_content,specific_cloud_liquid_water_content" sbatch run_eval.sbatch
# Re-runs reuse the per-variable truth caches + per-(var,level) CSVs, so only
# plotting repeats. NOTE: the per-run truth build is single-pass and writes its
# caches only at the end -- if a NEW multi-variable build is interrupted before
# finishing, no cache is saved and it restarts; hence the generous walltime.
#SBATCH --array=0-3
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
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
export EVAL_VARS="${EVAL_VARS:-}"   # empty -> eval_common DEFAULT_VARS (the 5 core)

RUNS=(nextgems2049 era5_1955 era5_2023 era5_2026)
export EVAL_RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID} run=${EVAL_RUN} "\
"interval=${NG_LEVEL_INTERVAL} levels=${NG_LEVELS:-<interval>} vars=${EVAL_VARS:-<default5>}"

# 1) build the per-variable model-grid truth caches once, single pass (no-op if present)
"$PY" -c "import eval_common as c; c.ensure_native_truth()"
# 2) all three plot families, each sweeping every requested variable x level
"$PY" plot_spaghetti.py
"$PY" plot_drift.py
"$PY" plot_drift_maps.py

echo "$(date '+%F %T') done run=${EVAL_RUN}"
