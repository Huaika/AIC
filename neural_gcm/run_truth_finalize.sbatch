#!/bin/bash
#SBATCH --job-name=truth_final
# STAGE 2: concatenate the per-chunk truth parts into the final per-variable
# caches (results_eval_<run>/truth_modelgrid_<SHORT>_<run>.nc). Cheap (small
# model-grid arrays). Only finalizes variables whose parts cover ALL timesteps.
#   EVAL_RUN exported by the submitter; typically run with afterok on the
#   matching truth-chunk array.
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:15:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/eval_logs/final_%x_%A.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/eval_logs/final_%x_%A.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/eval_logs

PY="$(pwd)/.venv/bin/python"
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export EVAL_VARS="${EVAL_VARS:-}"
: "${EVAL_RUN:?export EVAL_RUN before submitting}"

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID} run=${EVAL_RUN} finalize"
"$PY" -c "import eval_common as c; c.finalize_truth()"
echo "$(date '+%F %T') done finalize run=${EVAL_RUN}"
