#!/bin/bash
# Watch the first GraphCast rollout tasks that land on an A100. If one fails with
# a GPU out-of-memory (JAX RESOURCE_EXHAUSTED / CUDA OOM), fall back to H100-only:
#   - edit rollout_batch.sbatch partition -> gpu_h100_short (future submits)
#   - scontrol-update every queued gc-rollout array to gpu_h100_short (pending tasks)
#   - requeue the OOM-failed task
# If an A100 task COMPLETES fine, log that A100 works and exit (no change).
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-/pfs/data6/home/ka/ka_iti/ka_dm9435/code}"
cd "$REPO_ROOT"
U=$(whoami)
SBATCH=graphcast/inference/rollout_batch.sbatch
LOGDIR="$REPO_ROOT/logs"
OOM_RE='RESOURCE_EXHAUSTED|Resource exhausted|CUDA_ERROR_OUT_OF_MEMORY|out of memory|XlaRuntime'
CONFIRM_OK="${CONFIRM_OK:-1}"
seen_ok=0

log(){ echo "$(date '+%F %T') $*"; }

fallback_to_h100() {
  local failed_task="$1"
  log "A100 OOM detected on task $failed_task -> switching to H100-only"
  sed -i 's/^#SBATCH --partition=gpu_h100_short,gpu_a100_short/#SBATCH --partition=gpu_h100_short/' "$SBATCH"
  grep -m1 '^#SBATCH --partition=' "$SBATCH"
  for jid in $(squeue -u "$U" -h -o "%A|%j" 2>/dev/null | awk -F'|' '/gc-rollout-/{print $1}' | sort -u); do
    scontrol update jobid="$jid" Partition=gpu_h100_short 2>/dev/null && log "updated $jid -> gpu_h100_short"
  done
  scontrol requeue "$failed_task" 2>/dev/null && log "requeued $failed_task"
}

log "watching for first A100 rollout task outcome..."
while true; do
  while read -r jid part state; do
    [ -z "${jid:-}" ] && continue
    case "$part" in
      *a100*) : ;;
      *) continue ;;
    esac
    case "$state" in
      COMPLETED)
        seen_ok=$((seen_ok + 1))
        log "A100 task $jid COMPLETED ok ($seen_ok/$CONFIRM_OK)"
        if [ "$seen_ok" -ge "$CONFIRM_OK" ]; then
          log "A100 works -> no change needed, exiting"
          exit 0
        fi
        ;;
      FAILED|OUT_OF_MEMORY|NODE_FAIL)
        if grep -qiE "$OOM_RE" "$LOGDIR"/gc-rollout-*.out 2>/dev/null; then
          fallback_to_h100 "$jid"
          log "fallback done, exiting"
          exit 0
        else
          log "A100 task $jid $state but no OOM signature in logs - ignoring"
        fi
        ;;
    esac
  done < <(sacct -n -X --starttime now-1days --format=JobID,Partition,State \
             --name=gc-rollout-2023,gc-rollout-1955,gc-rollout-2049 2>/dev/null \
           | awk 'NF>=3{print $1, $2, $3}')
  sleep 120
done
