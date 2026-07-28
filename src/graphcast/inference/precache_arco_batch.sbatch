#!/bin/bash
#SBATCH --job-name=gc-precache-arco
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_logs_dm9435/%x-%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_logs_dm9435/%x-%A_%a.log
#SBATCH --array=0-354
# cpu_il (264 nodes) + cpu (80) are QoS 'normal' with no submit cap (the dev_*
# partitions cap at 4 submitted). Slurm packs several tasks per node; that shares
# the node NIC (~22 min/case) but schedules immediately -- the simple approach.
#SBATCH --partition=cpu_il,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# ~39GB actual peak; 64GB gives headroom (12GB/24GB OOMed). 3h walltime: packed
# cases share the NIC (~30-45 min, some >1h) -> 1h caused sporadic TIMEOUTs, so
# give ample margin. cpu/cpu_il allow up to 3 days.
#SBATCH --mem=64gb
#SBATCH --time=03:00:00
#
# Precache the GraphCast-ready ARCO ERA5 case NetCDFs ONCE into MY persistent
# workspace, so rollout jobs read them locally instead of streaming ARCO per run
# (bwUniCluster /scratch is node-local and wiped per job). GraphCast analogue of
# neural_gcm/stage_era5.py.
#
# Tuned for FAST scheduling via backfill: tiny 1-core / 30-min jobs, ONE
# init-day case per array task, submitted across several CPU partitions at once
# so Slurm starts each wherever a slot frees first. NetCDF zlib is single-
# threaded, and the cost is ARCO streaming (network-bound), so 1 core is enough
# and one case (~10-15 min) fits comfortably in 30 min. Resumable: a task whose
# case already exists+validates exits immediately.
#
# Full year 2023 (355 init-days -> 355 tasks):
#   sbatch graphcast/inference/precache_arco_batch.sbatch
# Fewer cases per task via COUNT (e.g. 2) with a smaller array:
#   sbatch --array=0-177 --export=ALL,COUNT=2 graphcast/inference/precache_arco_batch.sbatch
# Different year:
#   sbatch --export=ALL,YEAR=2026 graphcast/inference/precache_arco_batch.sbatch
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_logs_dm9435

PY="$REPO_ROOT/neural_gcm/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT/src/graphcast/inference"

export YEAR="${YEAR:-2023}"
export INIT_HOUR="${INIT_HOUR:-6}"
export STRIDE="${STRIDE:-1}"
export STEPS="${STEPS:-40}"
export STEP_HOURS="${STEP_HOURS:-6}"
export COUNT="${COUNT:-1}"                 # init-days per array task
export BUDGET_MIN="${BUDGET_MIN:-24}"      # stop starting new cases after this
export CACHE_DIR="${CACHE_DIR:-/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/graphcast_arco_cases}"
# Eager load (proven by the timing job). The build's selection of 42 x 13-level
# global 0.25deg frames peaks ~30-40GB, so give 64GB. (dask 'native' chunking did
# NOT keep memory low here -- it still peaked >23GB and OOMed at 24GB.)
export GRAPHCAST_ARCO_CHUNKS="${GRAPHCAST_ARCO_CHUNKS:-none}"
export TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$CACHE_DIR"

echo "$(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-manual} task=${TASK_ID} year=${YEAR} count=${COUNT} cache=${CACHE_DIR}"

exec "$PY" - <<'PY'
import os, time
from pathlib import Path
import pandas as pd
from arco_era5_graphcast_case import ensure_arco_graphcast_case, DEFAULT_ARCO_ERA5_PATH
from precache_arco_case import DEFAULT_PRESSURE_LEVELS, validation_error

year = int(os.environ["YEAR"]); hour = int(os.environ["INIT_HOUR"])
stride = int(os.environ["STRIDE"]); steps = int(os.environ["STEPS"]); step_h = int(os.environ["STEP_HOURS"])
count = int(os.environ["COUNT"]); budget_s = float(os.environ["BUDGET_MIN"]) * 60.0
cache = Path(os.environ["CACHE_DIR"]); cache.mkdir(parents=True, exist_ok=True)
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

t0 = time.time()
for it in mine:
    if time.time() - t0 > budget_s:
        print(f"budget reached, leaving remaining for a resubmit (resumable)", flush=True)
        break
    try:
        path = ensure_arco_graphcast_case(
            cache, it, steps, step_h, levels,
            arco_path=DEFAULT_ARCO_ERA5_PATH, overwrite=False, compression_level=1)
        err = validation_error(path, steps, levels)
        if err is not None:
            path = ensure_arco_graphcast_case(
                cache, it, steps, step_h, levels,
                arco_path=DEFAULT_ARCO_ERA5_PATH, overwrite=True, compression_level=1)
            err = validation_error(path, steps, levels)
        print(f"{'OK  ' if err is None else 'BAD '}{it} -> {path.name}"
              + ("" if err is None else f" ({err})"), flush=True)
    except Exception as e:
        print(f"FAIL {it}: {e}", flush=True)
        raise
print(f"task {task} done in {time.time()-t0:.0f}s", flush=True)
PY
