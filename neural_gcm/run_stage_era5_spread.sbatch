#!/bin/bash
#SBATCH --job-name=era5_stage_spread
#SBATCH --partition=cpu
# Re-stage the months that got dogpiled on one node in the first array run.
# ONE multi-node job with --ntasks-per-node=1: each month runs as its own srun
# step pinned (via step-level --exclusive) to a distinct node, so no two of MY
# tasks share a node -- without taking whole nodes exclusively from other users.
# 9 nodes process the 17 months; a node picks up the next month when its step
# finishes. Resumable: stage_era5.py skips a month whose .nc already exists.
#SBATCH --nodes=9
#SBATCH --ntasks=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=04:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_staging_logs/spread_%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_staging_logs/spread_%j.log
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
LOGDIR=/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_staging_logs
mkdir -p "$LOGDIR"
PY="$(pwd)/.venv/bin/python"
NODES=9

# months still needing download (year month) -- 1955-06 is handled by the
# still-running first-array task and is intentionally excluded.
MONTHS=( "1955 05" "1955 07" "1955 10" "1955 11" "1955 12" \
         "2023 01" "2023 02" "2023 03" "2023 04" "2023 05" "2023 06" \
         "2023 07" "2023 08" "2023 09" "2023 10" "2023 11" "2023 12" )

echo "$(date '+%F %T') spread job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}; "\
"${#MONTHS[@]} months over ${NODES} nodes"

for spec in "${MONTHS[@]}"; do
  set -- $spec; y=$1; m=$2
  ERA5_YEAR=$y ERA5_MONTH=$m srun --exclusive --ntasks=1 --nodes=1 \
      --output="$LOGDIR/spread_${y}_${m}.log" \
      "$PY" stage_era5.py &
  # keep at most NODES steps in flight -> one per node
  while [ "$(jobs -r | wc -l)" -ge "$NODES" ]; do sleep 5; done
done
wait
echo "$(date '+%F %T') spread job done"
