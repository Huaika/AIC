#!/bin/bash
#SBATCH --job-name=ngcm_plot
# STAGE 3: plotting, one 30-min dev_cpu job per VARIABLE (array index = variable).
# Each task runs all 3 plot families (spaghetti / drift-stat / drift-map) for its
# single variable, sweeping every requested level. Truth caches must already
# exist (EVAL_REQUIRE_CACHE=1 makes a job fail fast rather than attempt a
# multi-hour build). Per-(var) prediction scans are cached to CSV/nc, so a re-run
# only re-renders. ~14 min/variable observed -> fits 30 min.
#   EVAL_RUN exported by the submitter; typically afterok on the finalize job.
#SBATCH --partition=dev_cpu,dev_cpu_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=00:29:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/eval_logs/plot_%x_%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/eval_logs/plot_%x_%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/eval_logs

PY="$(pwd)/.venv/bin/python"
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NG_LEVEL_INTERVAL="${NG_LEVEL_INTERVAL:-50}"
export EVAL_REQUIRE_CACHE=1
: "${EVAL_RUN:?export EVAL_RUN before submitting}"

# array index selects ONE variable from the default-5 list (override via VARLIST)
VARLIST=(temperature geopotential specific_humidity u_component_of_wind v_component_of_wind)
export EVAL_VARS="${VARLIST[$SLURM_ARRAY_TASK_ID]}"

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID} run=${EVAL_RUN} "\
"var=${EVAL_VARS} levels=${NG_LEVELS:-<interval ${NG_LEVEL_INTERVAL}>}"

"$PY" plot_spaghetti.py
"$PY" plot_drift.py
"$PY" plot_drift_maps.py

echo "$(date '+%F %T') done plots run=${EVAL_RUN} var=${EVAL_VARS}"
