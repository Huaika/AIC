#!/bin/bash
#SBATCH --job-name=gc-gate
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_precache_logs/%x-%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_precache_logs/%x-%j.log
#SBATCH --partition=dev_cpu_il,dev_cpu,cpu_il,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=00:30:00
#
# SHORT one-shot heal gate. Launch it AFTER a precache array via a dependency so
# the waiting holds no node, e.g.:
#   PID=<precache array job id>
#   sbatch --job-name=gc-gate-2023 --dependency=afterany:$PID \
#     --export=ALL,YEAR=2023,CASE_MODE=arco,PRECACHE_JOBNAME=gc-precache-arco \
#     graphcast/inference/heal_gate.sbatch
# It validates+heals and then launches the rollout (or re-chains itself after a
# refill). Runs on the fast dev_cpu partitions since it's a ~2-min real-work job.
set -uo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
export REPO_ROOT
echo "$(date '+%F %T') host=$(hostname) gate YEAR=${YEAR} CASE_MODE=${CASE_MODE}"
exec bash graphcast/inference/heal_gate.sh
