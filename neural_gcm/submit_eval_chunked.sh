#!/bin/bash
# Orchestrate the chunked eval pipeline as 30-min dev_cpu jobs, one independent
# dependency chain per run:  truth-chunks (array)  ->  finalize  ->  plots (array)
# so fast runs (era5_2026) complete without waiting on slow ones (nextgems2049).
#
#   chunks : array 0..MAXCHUNK, each a CHUNK_STEPS time-slice; resumable
#   finalize: afterok on the chunk array; concatenates parts -> caches
#   plots  : afterok on finalize; array 0..4, one variable each
#
# Re-running is safe: completed parts/caches and cached CSVs are skipped.
# Usage:  bash submit_eval_chunked.sh
set -euo pipefail
cd "$(dirname "$0")"

RUNS=(nextgems2049 era5_1955 era5_2023 era5_2026)
MAXCHUNK=14          # 15 chunks x 100 steps = 1500 >= 1460 (max nsteps); extra are no-ops
export CHUNK_STEPS=100   # ~16 min/chunk worst case (era5, 37 lvl) -> margin under 29 min
export NG_LEVEL_INTERVAL="${NG_LEVEL_INTERVAL:-50}"

for run in "${RUNS[@]}"; do
  cj=$(sbatch --parsable --export=ALL,EVAL_RUN="$run" \
        --array=0-${MAXCHUNK} run_truth_chunks.sbatch)
  fj=$(sbatch --parsable --dependency=afterok:"$cj" --kill-on-invalid-dep=yes \
        --export=ALL,EVAL_RUN="$run" run_truth_finalize.sbatch)
  pj=$(sbatch --parsable --dependency=afterok:"$fj" --kill-on-invalid-dep=yes \
        --export=ALL,EVAL_RUN="$run" --array=0-4 run_plots.sbatch)
  echo "$run : chunks=$cj  finalize=$fj  plots=$pj"
done
