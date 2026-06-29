# NextGEMS-2049 — 6 h / 10-day NeuralGCM rollouts: how the data was generated

Handoff for another agent. This documents **which scripts produced the rollout
data** and how, so the run can be reproduced, resumed, or extended.

## What the data is
NeuralGCM 10-day forecasts at 6 h output, **one forecast per init-day** over the
year 2049, **initialised and forced by the NextGEMS-2049 climate simulation**
(not ERA5 — 2049 is a future-climate run, so there is no ERA5 "truth"). Each
output file holds the **full prediction fields** (all model variables, 37 levels,
41 lead steps = 0…240 h).

## Where the data lives
**Rollout outputs:** `/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/nextgems_2049/predictions/`
→ `pred_2049_<YYYY-MM-DD>.nc`, 355 files, ~260 MB each (~92 GB total).
(NB: `…/ka_hc5935-ai-climate/nextgems_2049/` holds only a *copy of a NextGEMS input*
file, not the rollouts.)

**Source NextGEMS data (read-only, owner ka_je2428):**
`/pfs/work9/workspace/scratch/ka_je2428-nextgems_2049/`
- `3D_nextgems_2049_6hourly_0.25deg_lat-lon.nc` — z,q,t,u,v,w,ciwc,clwc, 25 levels, 1460 6 h steps, 0.25°
- `surface_nextgems_2049_6hourly_0.25deg_ci_SSTs_lat-lon.nc` — sst, ci (the forcing)
- (the 644 GB plain-surface and precip files are NOT used)

## The scripts that generated it (all under `code/neural_gcm/`)

### 1. `setup_env.sh` — build the environment
Creates the pinned **Python 3.11 venv** `neural_gcm/.venv` (uv) with neuralgcm,
jax[cuda12] 0.10.1, dinosaur, gcsfs, xarray, zarr, nbconvert. Additionally
required and installed into the venv: **netCDF4, h5netcdf, scipy** (to read the
NextGEMS NetCDF files; the venv ships no pip → `./.venv/bin/python -m ensurepip
--upgrade` then `-m pip install netCDF4 h5netcdf scipy`).

### 2. `nextgems_2049_rollout.py` — THE generator (one init-day = one process)
Pipeline per init-day (mirrors the ERA5 `rackow_daily_rollouts.ipynb`, with
NextGEMS substituted for ERA5):
1. **Load** the model checkpoint `v1/deterministic_2_8_deg.pkl` (streamed from
   `gs://neuralgcm/models/`; compute nodes have outbound internet).
   - input vars: geopotential, specific_humidity, temperature,
     u/v_component_of_wind, specific_cloud_ice/liquid_water_content
   - forcing vars: sea_ice_cover, sea_surface_temperature
   - model grid: 128 lon × 64 lat; 37 ERA5 pressure levels
2. **Rename** NextGEMS vars → NeuralGCM names (z→geopotential, t→temperature,
   q→specific_humidity, u/v→*_component_of_wind, ciwc/clwc→specific_cloud_*_water_content,
   ci→sea_ice_cover, sst→sea_surface_temperature, lat/lon→latitude/longitude).
3. **Vertical-interpolate** the 25 native NextGEMS levels → the model's 37 ERA5
   levels (linear, extrapolate).
4. **Horizontal-regrid** 0.25° → model grid via dinosaur `ConservativeRegridder`.
5. **Forcing masking (important fix):** NextGEMS sst/ci encode land/under-ice
   points as non-physical junk (up to ~9999) and the bad footprint drifts across
   timesteps. Mask sst to (270,310) K and ci to [0,1] → NaN, then **regrid each
   forcing timestep independently** (a single time trivially satisfies the
   regridder's fixed-NaN-mask requirement). The 3D atmospheric inputs are clean.
6. **encode + unroll:** `model.encode(inputs, input_forcings, jax.random.key(42))`
   then `model.unroll(state, all_forcings, steps=41, timedelta=6h,
   start_with_input=True)` → 10-day forecast at 6 h (leads 0…240 h, 41 frames).
7. **Write** `model.data_to_xarray(...)` to one NetCDF per init-day, with
   `lead_hours` and `valid_time` coords and `init_date`/`model` attrs. Atomic
   `.tmp`→rename, float32+zlib. **Resumable:** an init-day whose `.nc` exists is
   skipped.

Fully **env-parameterised** (defaults in parentheses):
`NG_DATA_DIR`, `NG_OUT_DIR`, `NG_MODEL` (v1/deterministic_2_8_deg.pkl),
`NG_YEAR` (2049), `NG_ROLLOUT_DAYS` (10), `NG_OUT_H` (6), `NG_SST_STRIDE_H` (24),
`NG_INIT_STRIDE_DAYS` (1), `NG_SEED` (42). Which init-day this process runs:
`NG_INIT_DATE` (explicit) or `NG_INIT_INDEX` (0-based index into the year's
init-day list, used by the Slurm array). Init-days are capped so init+10 d stays
within the data → **355 init-days** (2049-01-01 … 2049-12-21), index 0…354.

### 3. `run_nextgems_2049.sbatch` — Slurm job array (the actual run)
`#SBATCH --array=0-354` (no concurrency cap), `--partition=gpu_h100_short,gpu_h100`,
1×H100, 16 cores, 64 GB, 25 min. Sets `NG_INIT_INDEX=$SLURM_ARRAY_TASK_ID`,
`JAX_PLATFORMS=cuda` (fail loudly if no GPU), `XLA_PYTHON_CLIENT_ALLOCATOR=default`.
Each task ran ~2 min, ~5 GB RAM. Submit / re-fill gaps:
```bash
cd /pfs/data6/home/ka/ka_iti/ka_dm9435/code/neural_gcm
sbatch run_nextgems_2049.sbatch          # all 355 (skips existing → resumable)
sbatch --array=0-9 run_nextgems_2049.sbatch   # subset
```

## Reproduce / extend
- **Re-run / fill gaps:** just `sbatch run_nextgems_2049.sbatch` (idempotent).
- **Different year/length/cadence:** set `NG_YEAR`, `NG_ROLLOUT_DAYS`, `NG_OUT_H`
  and re-size the array (`init-days = days_in_year - rollout_days`).
- **Single day, interactively:** `NG_INIT_DATE=2049-06-15 ./.venv/bin/python nextgems_2049_rollout.py`.

## Output file schema (`pred_2049_<date>.nc`)
dims `time=41` (lead 0…240 h @ 6 h), `level=37`, `longitude=128`, `latitude=64`;
data vars = all NeuralGCM prognostics (geopotential, temperature, u/v wind,
specific humidity, cloud ice/liquid water); coords `lead_hours`, `valid_time`;
attrs `init_date`, `model`, `rollout_days`, `output_hours`, `sst_stride_hours`, `seed`.

## Downstream (analysis only — NOT data generation)
The plots were built on top of these files by `ng2049_common.py` +
`nextgems_2049_spaghetti.py` / `nextgems_2049_drift.py` / `nextgems_2049_drift_maps.py`
(orchestrated by `run_nextgems_2049_plots.sbatch`). They are documented separately;
they do not produce the rollout data.
