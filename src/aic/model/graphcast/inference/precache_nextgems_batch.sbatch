#!/bin/bash
#SBATCH --job-name=gc-precache-nextgems
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_precache_logs/%x-%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_precache_logs/%x-%A_%a.log
#SBATCH --array=0-354
#SBATCH --partition=cpu_il,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --time=03:00:00
#
# Precache GraphCast-ready NextGEMS case NetCDFs (default year 2049) into MY
# workspace. Same idea as precache_arco_batch.sbatch but the source is the LOCAL
# NextGEMS files (no cloud streaming), via nextgems_graphcast_case. One init-day
# case per array task, resumable (valid existing case -> skip).
#
#   sbatch graphcast/inference/precache_nextgems_batch.sbatch
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_precache_logs

PY="$REPO_ROOT/neural_gcm/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT/graphcast/inference"

export YEAR="${YEAR:-2049}"
export INIT_HOUR="${INIT_HOUR:-6}"
export STRIDE="${STRIDE:-1}"
export STEPS="${STEPS:-40}"
export STEP_HOURS="${STEP_HOURS:-6}"
export COUNT="${COUNT:-1}"
export CACHE_DIR="${CACHE_DIR:-/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/graphcast_nextgems_cases}"
export NEXTGEMS_ROOT="${NEXTGEMS_ROOT:-/pfs/work9/workspace/scratch/ka_je2428-nextgems_${YEAR}}"
export TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$CACHE_DIR"

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-manual} task=${TASK_ID} year=${YEAR} root=${NEXTGEMS_ROOT} cache=${CACHE_DIR}"

exec "$PY" - <<'PY'
import os
from pathlib import Path
import pandas as pd
from nextgems_graphcast_case import ensure_nextgems_graphcast_case, default_case_path
from precache_arco_case import DEFAULT_PRESSURE_LEVELS, validation_error

year = int(os.environ["YEAR"]); hour = int(os.environ["INIT_HOUR"])
stride = int(os.environ["STRIDE"]); steps = int(os.environ["STEPS"]); step_h = int(os.environ["STEP_HOURS"])
count = int(os.environ["COUNT"])
cache = Path(os.environ["CACHE_DIR"]); cache.mkdir(parents=True, exist_ok=True)
root = Path(os.environ["NEXTGEMS_ROOT"]) if os.environ.get("NEXTGEMS_ROOT") else None
task = int(os.environ["TASK_ID"])
levels = tuple(int(x) for x in DEFAULT_PRESSURE_LEVELS)

lead = pd.Timedelta(hours=steps * step_h)
year_end = pd.Timestamp(f"{year}-12-31T23:00")
starts = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=f"{stride}D")
inits = [s + pd.Timedelta(hours=hour) for s in starts
         if (s + pd.Timedelta(hours=hour) + lead) <= year_end]
inits = [t.strftime("%Y-%m-%dT%H:00") for t in inits]

mine = inits[task * count:(task + 1) * count]
if not mine:
    print(f"task {task}: no init-days (of {len(inits)}); nothing to do.", flush=True)
    raise SystemExit(0)
print(f"task {task}: {len(mine)} init-day(s) of {len(inits)}: {mine}", flush=True)

for it in mine:
    try:
        path = ensure_nextgems_graphcast_case(
            cache, it, steps, step_h, levels, year=year, root=root,
            overwrite=False, compression_level=1)
        err = validation_error(path, steps, levels)
        if err is not None:
            path = ensure_nextgems_graphcast_case(
                cache, it, steps, step_h, levels, year=year, root=root,
                overwrite=True, compression_level=1)
            err = validation_error(path, steps, levels)
        print(f"{'OK  ' if err is None else 'BAD '}{it} -> {path.name}"
              + ("" if err is None else f" ({err})"), flush=True)
    except Exception as e:  # keep going: NextGEMS coverage may be partial
        print(f"FAIL {it}: {e}", flush=True)
print(f"task {task} done", flush=True)
PY
