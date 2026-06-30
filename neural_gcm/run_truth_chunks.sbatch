#!/bin/bash
#SBATCH --job-name=truth_chunk
# STAGE 1 of the chunked eval pipeline: build the per-variable model-grid truth
# caches in short (<=30 min) cpu-partition jobs -- short walltime is the backfill
# lever and the normal QOS has no concurrency cap (dev QOS is limited to 1 running
# / 4 queued, so NOT used). Each ARRAY TASK = one time-chunk of CHUNK_STEPS
# timesteps; it regrids every still-missing variable for that slice and writes a
# per-chunk part file (results_eval_<run>/truth_parts/<run>/). Resumable: existing
# parts/caches are skipped, so a re-submit only fills gaps.
#   EVAL_RUN must be exported by the submitter (per-run chains).
#   array index = chunk number; CHUNK_STEPS default 120 (=> <=13 chunks for 1460
#   steps). Tasks whose t0 >= nsteps are fast no-ops (e.g. era5_2026, 696 steps).
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24gb
#SBATCH --time=00:29:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/eval_logs/chunk_%x_%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/eval_logs/chunk_%x_%A_%a.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/eval_logs

PY="$(pwd)/.venv/bin/python"
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export EVAL_TRUTH_BATCH="${EVAL_TRUTH_BATCH:-12}"   # smaller slab -> lower memory
export EVAL_VARS="${EVAL_VARS:-}"                    # empty -> the 5 core vars
: "${EVAL_RUN:?export EVAL_RUN before submitting}"
CHUNK_STEPS="${CHUNK_STEPS:-120}"

idx=${SLURM_ARRAY_TASK_ID}
T0=$(( idx * CHUNK_STEPS ))
T1=$(( T0 + CHUNK_STEPS ))
echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID} run=${EVAL_RUN} "\
"chunk=${idx} steps=[${T0},${T1}) vars=${EVAL_VARS:-<default5>}"

"$PY" -c "import eval_common as c; c.build_truth_chunk(${T0}, ${T1})"

echo "$(date '+%F %T') done chunk ${idx} run=${EVAL_RUN}"
