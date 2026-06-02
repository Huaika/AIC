#!/usr/bin/env bash
#
# GPU-QUEUE scheduler for the NeuralGCM daily-rollout experiment.
#
#   - Runs at most ONE year per available GPU at a time.
#   - If there are fewer GPUs than years, the extra years WAIT IN A QUEUE and
#     start as GPUs free up. With a single GPU: year 1, then year 2, ...
#   - GPU-ONLY: a year is only ever started on a real GPU. If no GPU is present
#     (e.g. container restarted without one), the queue just waits and polls
#     until a GPU appears, then resumes.
#   - RESUMABLE: a year that crashes / is pre-empted is requeued and continues
#     from its last finished init-day (the notebook skips done inits). A year
#     that finishes is marked done (results_daily/.done_<year>) so re-running
#     this scheduler after a restart picks up only the unfinished years.
#
# Safe to run on every container start. Edit the QUEUE (priority order) or set
#   RACKOW_YEARS_QUEUE="1955 2023 2049"   ./run_all_daily.sh
set -uo pipefail
cd "$(dirname "$0")"

read -ra QUEUE <<< "${RACKOW_YEARS_QUEUE:-1955 2023}"
POLL="${RACKOW_POLL_SECONDS:-30}"
mkdir -p results_daily

# --- failsafe: rebuild the env if a container rebuild wiped it -----------------
if ! ./.venv/bin/python -c "import neuralgcm" >/dev/null 2>&1; then
  echo "$(date '+%F %T') venv missing/broken -> rebuilding via setup_env.sh ..."
  bash setup_env.sh
fi

list_gpus() {  # indices of GPUs visible right now (empty if none)
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' '
}

declare -A GPU_BUSY   # gpu_id -> pid
declare -A PID_YEAR   # pid    -> year
declare -A PID_GPU    # pid    -> gpu_id

# Working queue = requested years that are not already marked complete.
work=()
for y in "${QUEUE[@]}"; do
  [ -f "results_daily/.done_${y}" ] && { echo "$(date '+%F %T') year ${y} already done -> skip"; continue; }
  work+=("$y")
done
echo "$(date '+%F %T') queue (priority order): ${work[*]:-<none>}"

while :; do
  # 1) reap finished slots
  for pid in "${!PID_YEAR[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    wait "$pid"; rc=$?
    y="${PID_YEAR[$pid]}"; g="${PID_GPU[$pid]}"
    unset "GPU_BUSY[$g]" "PID_YEAR[$pid]" "PID_GPU[$pid]"
    if [ "$rc" -eq 0 ]; then
      touch "results_daily/.done_${y}"
      echo "$(date '+%F %T') year ${y} COMPLETE on GPU ${g} -> freeing GPU"
    else
      echo "$(date '+%F %T') year ${y} interrupted (rc=${rc}) on GPU ${g} -> requeue (will resume)"
      work=("$y" "${work[@]}")          # front of queue so it resumes promptly
    fi
  done

  # 2) done?
  if [ "${#work[@]}" -eq 0 ] && [ "${#PID_YEAR[@]}" -eq 0 ]; then
    echo "$(date '+%F %T') all years complete."
    break
  fi

  # 3) assign queued years to currently-free GPUs
  for g in $(list_gpus); do
    [ "${#work[@]}" -eq 0 ] && break
    [ -n "${GPU_BUSY[$g]:-}" ] && continue
    y="${work[0]}"; work=("${work[@]:1}")
    echo "$(date '+%F %T') start year ${y} on GPU ${g}  (log: results_daily/run_${y}.log)"
    CUDA_VISIBLE_DEVICES="$g" nohup ./run_year_once.sh "$y" "$g" \
      >>"results_daily/run_${y}.log" 2>&1 &
    pid=$!
    GPU_BUSY[$g]=$pid; PID_YEAR[$pid]=$y; PID_GPU[$pid]=$g
  done

  # 4) status + wait
  running=""
  for pid in "${!PID_YEAR[@]}"; do running+="${PID_YEAR[$pid]}@gpu${PID_GPU[$pid]} "; done
  echo "$(date '+%F %T') running: ${running:-none} | queued: ${work[*]:-none}"
  sleep "$POLL"
done
