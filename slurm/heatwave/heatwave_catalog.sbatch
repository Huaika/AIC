#!/bin/bash
#SBATCH --job-name=hw_catalog
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs/catalog_%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs/catalog_%j.log
set -euo pipefail
cd /pfs/data6/home/ka/ka_iti/ka_dm9435/code
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs
PY="/pfs/data6/home/ka/ka_iti/ka_dm9435/code/neural_gcm/.venv/bin/python"
export HW_REGIONS="${HW_REGIONS:-europe north_america asia africa oceania south_america world}"
export HW_Q="${HW_Q:-0.95}"; export HW_WINDOW="${HW_WINDOW:-5}"
export HW_DATA_DIR="/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave"
export HW_CATALOG_DIR="/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_catalog"
exec "$PY" -m aic.controller.heatwave
