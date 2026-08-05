#!/bin/bash
# Templated Slurm submitter for aic analysis jobs.
#
# One place for the sbatch header + logging, so jobs stop copy-pasting #SBATCH
# blocks and hardcoded paths. It submits your command via `sbatch --wrap` and prints
# the job id.
#
# Usage:
#   slurm/submit.sh --name drift-maps --time 02:00:00 --mem 48gb --cpus 8 \
#       [--array 0-3] [--partition cpu] -- \
#       env EVAL_SOURCES=neuralgcm,graphcast EVAL_YEAR=2023 \
#           python -m aic.view.drift_maps
#
# Everything after `--` is the command to run (env assignments allowed).
#
# Defaults come from the environment so nothing cluster-specific is baked in:
#   AIC_PARTITION (default cpu)   AIC_LOG_DIR (default <repo>/logs)
#   AIC_ACCOUNT   (optional -> --account)
set -euo pipefail

NAME="aic"; TIME="01:00:00"; MEM="16gb"; CPUS="4"; ARRAY=""; NODES=1
PARTITION="${AIC_PARTITION:-cpu}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${AIC_LOG_DIR:-$REPO/logs}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)      NAME="$2"; shift 2 ;;
    --time)      TIME="$2"; shift 2 ;;
    --mem)       MEM="$2"; shift 2 ;;
    --cpus)      CPUS="$2"; shift 2 ;;
    --nodes)     NODES="$2"; shift 2 ;;
    --array)     ARRAY="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --)          shift; break ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "error: no command given (put it after '--')" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
CMD="$*"
declare -a OPTS=(
  --job-name "$NAME" --partition "$PARTITION"
  --nodes "$NODES" --ntasks 1 --cpus-per-task "$CPUS" --mem "$MEM" --time "$TIME"
  --output "$LOG_DIR/${NAME}_%A_%a.log" --error "$LOG_DIR/${NAME}_%A_%a.log"
)
[[ -n "$ARRAY" ]] && OPTS+=(--array "$ARRAY")
[[ -n "${AIC_ACCOUNT:-}" ]] && OPTS+=(--account "$AIC_ACCOUNT")

echo "submitting '$NAME' [$PARTITION, ${CPUS}c ${MEM} ${TIME}${ARRAY:+ array=$ARRAY}]" >&2
echo "  cmd: $CMD" >&2
JID=$(sbatch --parsable "${OPTS[@]}" --wrap "cd '$REPO' && $CMD")
echo "$JID"
