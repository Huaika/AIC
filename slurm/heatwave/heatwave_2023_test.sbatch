#!/bin/bash
#SBATCH --job-name=hw_2023
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=00:40:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs/test2023_%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs/test2023_%j.log
set -euo pipefail
cd /pfs/data6/home/ka/ka_iti/ka_dm9435/code
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs
PY="/pfs/data6/home/ka/ka_iti/ka_dm9435/code/neural_gcm/.venv/bin/python"
export HW_DATA_DIR="/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave"
exec "$PY" -m aic.controller.heatwave_2023_test
