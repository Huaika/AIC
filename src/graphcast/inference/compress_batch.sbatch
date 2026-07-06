#!/bin/bash
#SBATCH --job-name=gc-compress
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_precache_logs/%x-%A_%a.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/gc_precache_logs/%x-%A_%a.log
# 4 results/zlib-task (~4x347s = ~23min). Ideally one-per-result, but the QOS
# per-user submit limit rejects a 355-element array alongside the rollout arrays,
# so we batch 4/task -> 89 elements (fits the limit). COUNT=1 + --array=0-354 for
# strict one-per-result if the QOS limit is raised.
#SBATCH --array=0-88
# cpu_il/cpu allow the 1h walltime (dev_cpu* cap at 30min -> would reject --time=1h).
#SBATCH --partition=cpu_il,cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24gb
#SBATCH --time=01:00:00
#
# Stage 2 of the two-stage rollout: compress the raw (uncompressed) predictions
# written by the GPU rollout (RAW_MODE) into the final zlib NetCDFs, then delete
# the raw. One array task per result (~6 min: read ~15s + zlib ~330s). Resumable:
# skips a day whose final exists; skips (waits) if its raw isn't there yet.
#
#   sbatch --job-name=gc-compress-2023 --export=ALL,YEAR=2023,CASE_MODE=arco     compress_batch.sbatch
#   sbatch --job-name=gc-compress-2049 --export=ALL,YEAR=2049,CASE_MODE=nextgems compress_batch.sbatch
set -uo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "$REPO_ROOT"
PY="$REPO_ROOT/neural_gcm/.venv/bin/python"
export PYTHONPATH="$REPO_ROOT/graphcast/inference"
WS=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate

export YEAR CASE_MODE
export INIT_HOUR="${INIT_HOUR:-6}" STRIDE="${STRIDE:-1}" STEPS="${STEPS:-40}" STEP_HOURS="${STEP_HOURS:-6}"
export COUNT="${COUNT:-4}" COMPLEVEL="${COMPLEVEL:-4}"
export TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${CASE_MODE:-arco}" == "nextgems" ]]; then TAG="nextgems_${YEAR}"; else TAG="era5_${YEAR}"; fi
export RAW_DIR="$WS/graphcast_${TAG}/predictions_raw"
export FINAL_DIR="$WS/graphcast_${TAG}/predictions"

echo "$(date '+%F %T') host=$(hostname) compress YEAR=$YEAR tag=$TAG task=$TASK_ID"

exec "$PY" - <<'PY'
import os, subprocess, sys
from pathlib import Path
import pandas as pd

year=int(os.environ["YEAR"]); hour=int(os.environ["INIT_HOUR"]); stride=int(os.environ["STRIDE"])
steps=int(os.environ["STEPS"]); step_h=int(os.environ["STEP_HOURS"])
count=int(os.environ["COUNT"]); task=int(os.environ["TASK_ID"]); complevel=int(os.environ["COMPLEVEL"])
raw_dir=Path(os.environ["RAW_DIR"]); final_dir=Path(os.environ["FINAL_DIR"])
final_dir.mkdir(parents=True, exist_ok=True)

lead=pd.Timedelta(hours=steps*step_h); year_end=pd.Timestamp(f"{year}-12-31T23:00")
starts=pd.date_range(f"{year}-01-01", f"{year}-12-31", freq=f"{stride}D")
inits=[s+pd.Timedelta(hours=hour) for s in starts if (s+pd.Timedelta(hours=hour)+lead)<=year_end]
dates=[t.strftime("%Y-%m-%d") for t in inits]
mine=dates[task*count:(task+1)*count]
if not mine:
    print(f"task {task}: nothing to do"); sys.exit(0)

for d in mine:
    raw=raw_dir/f"pred_{year}_{d}.nc"; fin=final_dir/f"pred_{year}_{d}.nc"
    if fin.exists():
        print(f"SKIP {d} (final exists)", flush=True); continue
    if not raw.exists():
        print(f"WAIT {d} (raw not ready)", flush=True); continue
    rc=subprocess.call([sys.executable,"graphcast/inference/compress_pred.py",str(raw),str(fin),str(complevel)])
    if rc==0 and fin.exists():
        try: raw.unlink()
        except OSError: pass
        print(f"OK {d} -> {fin.name} (raw deleted)", flush=True)
    elif rc==3:
        print(f"NAN-SKIP {d} (all-NaN raw kept for investigation)", flush=True)
    else:
        print(f"FAIL {d} (rc={rc})", flush=True)
PY
