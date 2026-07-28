#!/bin/bash
# Orchestrator: wait for a precache array to drain, validate all cases for
# YEAR/CASE_MODE (auto-heal: delete corrupt + refill up to 4 rounds), then submit
# the rollout. Never launches the rollout unless every case validates. Meant to
# run inside a small CPU Slurm job (autostart_batch.sbatch), not on the login node.
#
# Inputs (simple, comma-free -- everything else is derived):
#   YEAR              e.g. 2023 | 1955 | 2049
#   CASE_MODE         arco | nextgems
#   PRECACHE_JOBNAME  (optional) name of the precache array to wait on; default
#                     gc-precache-<YEAR> (arco) / gc-precache-nextgems (nextgems)
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"
U=$(whoami)
PY="$REPO_ROOT/neural_gcm/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT/graphcast/inference"
WS=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate

: "${YEAR:?set YEAR}"; : "${CASE_MODE:?set CASE_MODE}"
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
ROLLOUT_JOBNAME="gc-rollout-$YEAR"

wait_precache() { while squeue -u "$U" -h -o "%j" 2>/dev/null | grep -qx "$PRECACHE_JOBNAME"; do sleep 60; done; }
submit_precache() {
  for i in $(seq 1 20); do
    squeue -u "$U" -h -o "%j" 2>/dev/null | grep -qx "$PRECACHE_JOBNAME" && return 0
    timeout 45 sbatch --parsable --job-name="$PRECACHE_JOBNAME" --export="ALL,$PRECACHE_EXPORT" "$PRECACHE_SBATCH" >/dev/null 2>&1 && { echo "resubmitted $PRECACHE_JOBNAME"; return 0; }
    sleep 15
  done
}
submit_rollout() {
  for i in $(seq 1 20); do
    squeue -u "$U" -h -o "%j" 2>/dev/null | grep -qx "$ROLLOUT_JOBNAME" && { echo "$ROLLOUT_JOBNAME already queued"; return 0; }
    if J=$(timeout 45 sbatch --parsable --job-name="$ROLLOUT_JOBNAME" --export="ALL,$ROLLOUT_EXPORT" "$ROLLOUT_SBATCH" 2>/dev/null); then
      echo "ROLLOUT SUBMITTED ($ROLLOUT_JOBNAME): $J"; return 0
    fi
    sleep 15
  done
  echo "ERROR: could not submit $ROLLOUT_JOBNAME"
}

echo "$(date '+%T') orchestrating YEAR=$YEAR CASE_MODE=$CASE_MODE precache=$PRECACHE_JOBNAME rollout=$ROLLOUT_JOBNAME cache=$CACHE_DIR"
for round in 1 2 3 4; do
  echo "$(date '+%T') [$YEAR/$CASE_MODE] round $round: waiting for $PRECACHE_JOBNAME to drain..."
  wait_precache
  if DELETE_INVALID=1 "$PY" graphcast/inference/validate_cases.py; then
    echo "$(date '+%T') [$YEAR/$CASE_MODE] ALL cases VALID (round $round)"; break
  fi
  echo "$(date '+%T') [$YEAR/$CASE_MODE] gaps remain -> resubmit precache"
  submit_precache
  sleep 30
done

echo "$(date '+%T') [$YEAR/$CASE_MODE] final validation:"
if DELETE_INVALID=0 "$PY" graphcast/inference/validate_cases.py; then
  echo "$(date '+%T') [$YEAR/$CASE_MODE] passed -> launching rollout"
  submit_rollout
else
  echo "$(date '+%T') [$YEAR/$CASE_MODE] STILL INCOMPLETE -> NOT launching rollout"
fi
