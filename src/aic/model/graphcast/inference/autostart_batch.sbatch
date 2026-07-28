#!/bin/bash
#SBATCH --job-name=gc-autostart
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_precache_logs/%x-%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_precache_logs/%x-%j.log
#SBATCH --partition=cpu_il,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=1-00:00:00
#
# Runs graphcast/inference/autostart.sh (wait->validate/auto-heal->rollout) off
# the login node. autostart.sh derives everything from YEAR+CASE_MODE, so only
# comma-free values are passed (avoids the nested-comma trap in --export). E.g.:
#
#   sbatch --job-name=gc-autostart-2049 \
#     --export=ALL,YEAR=2049,CASE_MODE=nextgems,PRECACHE_JOBNAME=gc-precache-nextgems \
#     graphcast/inference/autostart_batch.sbatch
#   sbatch --job-name=gc-autostart-1955 \
#     --export=ALL,YEAR=1955,CASE_MODE=arco,PRECACHE_JOBNAME=gc-precache-1955 \
#     graphcast/inference/autostart_batch.sbatch
set -uo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
export REPO_ROOT
echo "$(date '+%F %T') host=$(hostname) autostart YEAR=${YEAR} CASE_MODE=${CASE_MODE} precache=${PRECACHE_JOBNAME:-auto}"
exec bash graphcast/inference/autostart.sh
