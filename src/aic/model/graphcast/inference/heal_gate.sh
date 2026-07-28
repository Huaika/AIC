#!/bin/bash
# One-shot heal gate (runs as a SHORT cpu job, launched via --dependency=afterany
# on the precache array -- so the "waiting" holds no node). Does the real work:
# validate all cases for YEAR/CASE_MODE, delete corrupt ones, and then:
#   - all valid  -> launch the rollout (with an A100 probe + dependency fallback)
#   - gaps remain-> resubmit the precache array and re-chain another gate after it
# No polling loops; Slurm dependencies do the waiting.
#
# Inputs (comma-free): YEAR, CASE_MODE(arco|nextgems), PRECACHE_JOBNAME(optional)
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"
PY="$REPO_ROOT/neural_gcm/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT/graphcast/inference"
WS=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate
: "${YEAR:?}"; : "${CASE_MODE:?}"
export YEAR CASE_MODE
export INIT_HOUR="${INIT_HOUR:-6}" STRIDE="${STRIDE:-1}" STEPS="${STEPS:-40}" STEP_HOURS="${STEP_HOURS:-6}"

if [[ "$CASE_MODE" == "nextgems" ]]; then
  export CACHE_DIR="$WS/graphcast_nextgems_cases"
  PRECACHE_SBATCH=graphcast/inference/precache_nextgems_batch.sbatch
  PRECACHE_JOBNAME="${PRECACHE_JOBNAME:-gc-precache-nextgems}"
  PRECACHE_EXPORT="YEAR=$YEAR"
  ROLLOUT_EXPORT="GRAPHCAST_YEAR=$YEAR,GRAPHCAST_USE_NEXTGEMS=1,GRAPHCAST_NEXTGEMS_YEAR=$YEAR"
else
  export CACHE_DIR="$WS/graphcast_arco_cases"
  PRECACHE_SBATCH=graphcast/inference/precache_arco_batch.sbatch
  PRECACHE_JOBNAME="${PRECACHE_JOBNAME:-gc-precache-$YEAR}"
  PRECACHE_EXPORT="YEAR=$YEAR"
  ROLLOUT_EXPORT="GRAPHCAST_YEAR=$YEAR"
fi
ROLLOUT_SBATCH=graphcast/inference/rollout_batch.sbatch
GATE_SBATCH=graphcast/inference/heal_gate.sbatch
log(){ echo "$(date '+%F %T') [$YEAR/$CASE_MODE] $*"; }
S(){ timeout 60 sbatch "$@" 2>/dev/null; }   # sbatch with a bounded retry

launch_rollout() {
  # Two-stage, H100-only (A100-short 40GB OOMs, verified): the GPU rollout writes
  # RAW (uncompressed) predictions, then a CPU compress array (afterany) turns them
  # into the final zlib NetCDFs. rollout_batch.sbatch defaults to RAW_MODE=1.
  local roll
  roll=$(S --parsable --job-name="gc-rollout-$YEAR" --export="ALL,$ROLLOUT_EXPORT" "$ROLLOUT_SBATCH")
  if [[ -z "$roll" ]]; then log "rollout submit failed"; return 1; fi
  log "GPU rollout (raw) = $roll"
  S --parsable --job-name="gc-compress-$YEAR" --dependency="afterany:$roll" \
    --export="ALL,YEAR=$YEAR,CASE_MODE=$CASE_MODE" graphcast/inference/compress_batch.sbatch \
    | xargs -r -I{} echo "$(date '+%T') CPU compress stage (after $roll) = {}"
}

GATE_ROUND="${GATE_ROUND:-1}"
GATE_MAX_ROUNDS="${GATE_MAX_ROUNDS:-4}"
log "gate round $GATE_ROUND/$GATE_MAX_ROUNDS; validating cases in $CACHE_DIR"
if DELETE_INVALID=1 "$PY" graphcast/inference/validate_cases.py; then
  log "all cases valid -> launching rollout"
  launch_rollout
elif [[ "$GATE_ROUND" -ge "$GATE_MAX_ROUNDS" ]]; then
  log "gaps remain after $GATE_ROUND rounds -> launching rollout anyway (missing cases build inline on GPU)"
  launch_rollout
else
  log "gaps remain -> resubmit precache + re-chain gate (round $((GATE_ROUND+1)))"
  pid=$(S --parsable --job-name="$PRECACHE_JOBNAME" --export="ALL,$PRECACHE_EXPORT" "$PRECACHE_SBATCH")
  if [[ -n "$pid" ]]; then
    log "resubmitted precache = $pid"
    S --job-name="gc-gate-$YEAR" --dependency="afterany:$pid" \
      --export="ALL,YEAR=$YEAR,CASE_MODE=$CASE_MODE,PRECACHE_JOBNAME=$PRECACHE_JOBNAME,GATE_ROUND=$((GATE_ROUND+1))" "$GATE_SBATCH" \
      && log "re-chained gate after $pid"
  else
    log "ERROR: could not resubmit precache"
  fi
fi
