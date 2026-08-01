#!/bin/bash
#SBATCH --job-name=hw_anim
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=01:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs/anim_%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs/anim_%j.log
set -euo pipefail
cd /pfs/data6/home/ka/ka_iti/ka_dm9435/code
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs
PY="/pfs/data6/home/ka/ka_iti/ka_dm9435/code/neural_gcm/.venv/bin/python"
export JAX_PLATFORMS=cpu       # regridder loads the 2.8deg checkpoint on CPU
export HW_DATA_DIR="/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave"
export HW_CLIM_DIR="/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_clim"     # cached day-of-year threshold/mean/std
export HW_GIF_DIR="/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_gifs"
export HW_WINDOW="${HW_WINDOW:-5}"             # best-performing window
export HW_SPEC_REGION="${HW_SPEC_REGION:-europe}"
export HW_YEAR="${HW_YEAR:-2023}"
export HW_COVER_FRAC="${HW_COVER_FRAC:-0.10}"   # min region-area fraction/day
exec "$PY" -m aic.view.heatwave_anim
