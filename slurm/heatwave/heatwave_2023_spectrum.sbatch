#!/bin/bash
#SBATCH --job-name=hw_spectrum
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs/spectrum_%j.log
#SBATCH --error=/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs/spectrum_%j.log
set -euo pipefail
cd /pfs/data6/home/ka/ka_iti/ka_dm9435/code
mkdir -p /pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_logs
PY="/pfs/data6/home/ka/ka_iti/ka_dm9435/code/neural_gcm/.venv/bin/python"
export JAX_PLATFORMS=cpu            # regridder loads the NeuralGCM checkpoint (JAX) on CPU
export HW_DATA_DIR="/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave"
export HW_FIG_DIR="/pfs/data6/home/ka/ka_iti/ka_dm9435/code/outputs/figures/heatwave2023"
exec "$PY" -m aic.controller.heatwave_2023_spectrum
