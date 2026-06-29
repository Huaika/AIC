#!/usr/bin/env python
"""NeuralGCM daily 10-day/6 h rollouts driven by NextGEMS-2049 (not ERA5).

This is the NextGEMS analogue of ``rackow_daily_rollouts.ipynb``. Because 2049 is
a *future-climate* run there is no ERA5 "truth", so we do NOT compute RMSE-vs-ERA5;
each init-day's run instead writes the **full prediction fields** (all model output
variables / levels / lead times) as one NetCDF file.

Data path (mirrors the ERA5 notebook, with NextGEMS substituted for the source):
  load NextGEMS  ->  rename vars to NeuralGCM names  ->  vertical-interp the 25
  native levels onto NeuralGCM's 37 ERA5 levels  ->  ConservativeRegridder onto
  the model's horizontal grid  ->  model.encode + model.unroll.

One run == one init-day == one 10-day/6 h forecast. Designed to be one Slurm
array task per init-day (see run_nextgems_2049.sbatch). Resumable: an init-day
whose output NetCDF already exists is skipped.

Parameterisation (all via environment variables, like the rackow scripts):
  NG_DATA_DIR        NextGEMS data dir (default: ka_je2428 workspace)
  NG_OUT_DIR         output dir for prediction NetCDFs
  NG_MODEL           checkpoint name under gs://neuralgcm/models/
  NG_YEAR            year label (default 2049)
  NG_ROLLOUT_DAYS    rollout length in days (default 10)
  NG_OUT_H           output cadence in hours (default 6)
  NG_SST_STRIDE_H    prescribed-SST/ci sampling stride in hours (default 24)
  NG_INIT_STRIDE_DAYS  spacing between init-days in the year (default 1)
  NG_SEED            PRNG seed for encode (default 42)

Which init-day THIS process runs (pick exactly one):
  NG_INIT_DATE       explicit YYYY-MM-DD, OR
  NG_INIT_INDEX      0-based index into the year's init-day list (Slurm array id)

If neither is set, runs ALL init-days in the year sequentially (workstation mode).
"""
from __future__ import annotations

import os
from pathlib import Path

# Disable JAX's 75% pre-grab and keep the default (BFC) allocator: over a long
# loop BFC pools/reuses the working set instead of churning it (matches notebook).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default")

import pickle

import gcsfs
import jax
import numpy as np
import pandas as pd
import xarray

from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
import neuralgcm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = Path(os.environ.get(
    "NG_DATA_DIR",
    "/pfs/work9/workspace/scratch/ka_je2428-nextgems_2049",
))
OUT_DIR = Path(os.environ.get(
    "NG_OUT_DIR",
    "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/nextgems_2049/predictions",
))
MODEL_NAME = os.environ.get("NG_MODEL", "v1/deterministic_2_8_deg.pkl")
YEAR = os.environ.get("NG_YEAR", "2049")
ROLLOUT_DAYS = int(os.environ.get("NG_ROLLOUT_DAYS", "10"))
OUT_H = int(os.environ.get("NG_OUT_H", "6"))
SST_STRIDE_H = int(os.environ.get("NG_SST_STRIDE_H", "24"))
INIT_STRIDE_DAYS = int(os.environ.get("NG_INIT_STRIDE_DAYS", "1"))
SEED = int(os.environ.get("NG_SEED", "42"))

# NextGEMS native variable names -> NeuralGCM names (only renames present keys).
RENAME = {
    "z": "geopotential",
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "ciwc": "specific_cloud_ice_water_content",
    "clwc": "specific_cloud_liquid_water_content",
    "ci": "sea_ice_cover",
    "sst": "sea_surface_temperature",
    "lat": "latitude",
    "lon": "longitude",
}

# Physical-validity windows for the NextGEMS forcing fields. The source data
# encodes land / under-ice points as non-physical junk (a spread of large
# values up to ~9999) AND the bad footprint drifts slightly across timesteps, so
# we mask out-of-range values to NaN per timestep and let the regridder's
# nearest-neighbour fill replace them with the nearest valid ocean value.
SST_MIN_K, SST_MAX_K = 270.0, 310.0          # seawater freezing .. warmest ocean
CI_MIN, CI_MAX = 0.0, 1.0001                 # sea-ice fraction (0..1)

# NeuralGCM's 37 ERA5 pressure levels (hPa). NextGEMS ships 25 native levels;
# we linearly interpolate onto these (the model's expected vertical coordinate).
LEVELS_37 = np.array([
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000,
])

F_3D = DATA_DIR / "3D_nextgems_2049_6hourly_0.25deg_lat-lon.nc"
F_SST = DATA_DIR / "surface_nextgems_2049_6hourly_0.25deg_ci_SSTs_lat-lon.nc"


def _rename(ds: xarray.Dataset) -> xarray.Dataset:
    return ds.rename({k: v for k, v in RENAME.items() if k in ds.variables})


# --------------------------------------------------------------------------- #
# Model + regridder (built once per process)
# --------------------------------------------------------------------------- #
def load_model() -> neuralgcm.PressureLevelModel:
    gcs = gcsfs.GCSFileSystem(token="anon")
    with gcs.open(f"gs://neuralgcm/models/{MODEL_NAME}", "rb") as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)


def build_regridder(sample: xarray.Dataset, model) -> horizontal_interpolation.ConservativeRegridder:
    """ConservativeRegridder from the NextGEMS source grid to the model grid."""
    src_grid = spherical_harmonic.Grid(
        latitude_nodes=sample.sizes["latitude"],
        longitude_nodes=sample.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(sample.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(sample.longitude),
    )
    return horizontal_interpolation.ConservativeRegridder(
        src_grid, model.data_coords.horizontal, skipna=True
    )


# --------------------------------------------------------------------------- #
# Per-init-day forecast
# --------------------------------------------------------------------------- #
def _end_str(init_date: str, days: int) -> str:
    return str(np.datetime64(init_date) + np.timedelta64(days, "D"))


def run_one(model, regridder, ds3, dss, init_date: str) -> xarray.Dataset:
    """Encode at t0 and unroll a ROLLOUT_DAYS/OUT_H forecast for one init-day."""
    def regrid(ds):
        return xarray_utils.fill_nan_with_nearest(xarray_utils.regrid(ds, regridder))

    def regrid_per_time(ds):
        # fill_nan_with_nearest requires the NaN mask to be identical across
        # non-spatial dims; the NextGEMS forcing land/ice mask drifts across
        # time, so regrid each timestep on its own (a single time trivially
        # satisfies that) and re-concatenate.
        slabs = [regrid(ds.isel(time=[i])) for i in range(ds.sizes["time"])]
        return xarray.concat(slabs, dim="time")

    # --- atmospheric inputs at the single initial time t0 ---
    t0 = ds3[model.input_variables].sel(time=init_date, method="nearest").compute()
    t0 = t0.interp(level=LEVELS_37, method="linear",
                   kwargs={"fill_value": "extrapolate"})
    if "time" not in t0.dims:
        t0 = t0.expand_dims(time=[t0.time.values])
    t0_rg = regrid(t0)

    # --- prescribed, time-varying SST / sea-ice forcing over the rollout ---
    forcing = (
        dss[model.forcing_variables]
        .sel(time=slice(init_date, _end_str(init_date, ROLLOUT_DAYS)))
        .compute()
    )
    # subsample to the requested SST stride (nearest-in-time is used by unroll)
    step = max(1, SST_STRIDE_H // 6)  # native cadence is 6 h
    forcing = forcing.isel(time=slice(None, None, step))
    # mask non-physical land/under-ice junk -> NaN (regrid fills from nearest ocean)
    sst = forcing["sea_surface_temperature"]
    ci = forcing["sea_ice_cover"]
    forcing["sea_surface_temperature"] = sst.where((sst > SST_MIN_K) & (sst < SST_MAX_K))
    forcing["sea_ice_cover"] = ci.where((ci >= CI_MIN) & (ci <= CI_MAX)).clip(0.0, 1.0)
    forcing_rg = regrid_per_time(forcing)

    # --- encode + unroll ---
    steps = ROLLOUT_DAYS * 24 // OUT_H + 1          # +1 so leads run 0..days
    lead_h = np.arange(steps) * OUT_H
    timedelta = np.timedelta64(1, "h") * OUT_H

    inputs = model.inputs_from_xarray(t0_rg.isel(time=0))
    input_forcings = model.forcings_from_xarray(forcing_rg.isel(time=0))
    initial_state = model.encode(inputs, input_forcings, jax.random.key(SEED))

    all_forcings = model.forcings_from_xarray(forcing_rg)
    _, predictions = model.unroll(
        initial_state, all_forcings,
        steps=steps, timedelta=timedelta, start_with_input=True,
    )

    pred = model.data_to_xarray(predictions, times=lead_h)
    # attach physically meaningful time coordinates for plotting
    init_dt = np.datetime64(init_date)
    valid_time = init_dt + (lead_h * np.timedelta64(1, "h"))
    pred = pred.assign_coords(
        lead_hours=("time", lead_h.astype(np.int32)),
        valid_time=("time", valid_time),
    )
    pred.attrs.update(
        source="NeuralGCM rollout forced by NextGEMS-2049",
        model=MODEL_NAME,
        init_date=str(np.datetime64(init_date, "D")),
        rollout_days=ROLLOUT_DAYS,
        output_hours=OUT_H,
        sst_stride_hours=SST_STRIDE_H,
        seed=SEED,
    )
    return pred


# --------------------------------------------------------------------------- #
# Init-day selection
# --------------------------------------------------------------------------- #
def init_dates_for_year(ds3) -> list[str]:
    """All init-days in the year, capped so init + ROLLOUT_DAYS stays in range."""
    last = pd.Timestamp(np.datetime64(ds3.time.values[-1]))
    latest_init = last - pd.Timedelta(days=ROLLOUT_DAYS)
    dates = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31",
                          freq=f"{INIT_STRIDE_DAYS}D")
    dates = [d for d in dates if d <= latest_init]
    return [str(d.date()) for d in dates]


def out_path(init_date: str) -> Path:
    return OUT_DIR / f"pred_{YEAR}_{init_date}.nc"


def process(model, regridder, ds3, dss, init_date: str) -> None:
    out = out_path(init_date)
    if out.exists():
        print(f"[{init_date}] already done -> {out.name}, skip", flush=True)
        return
    print(f"[{init_date}] running {ROLLOUT_DAYS}-day/{OUT_H}h rollout ...", flush=True)
    pred = run_one(model, regridder, ds3, dss, init_date)
    tmp = out.with_suffix(".nc.tmp")
    # f4 keeps files compact; full fields on the 128x64 model grid are small.
    enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4}
           for v in pred.data_vars}
    pred.to_netcdf(tmp, encoding=enc)
    tmp.rename(out)  # atomic publish so a killed job never leaves a half file
    print(f"[{init_date}] wrote {out.name} "
          f"({out.stat().st_size / 1e6:.1f} MB, vars={list(pred.data_vars)})",
          flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("JAX devices:", jax.devices(), flush=True)

    model = load_model()
    print("input_variables  :", model.input_variables, flush=True)
    print("forcing_variables:", model.forcing_variables, flush=True)

    ds3 = _rename(xarray.open_dataset(F_3D, chunks={}))
    dss = _rename(xarray.open_dataset(F_SST, chunks={}))

    all_dates = init_dates_for_year(ds3)

    explicit = os.environ.get("NG_INIT_DATE", "").strip()
    idx_env = os.environ.get("NG_INIT_INDEX", "").strip()
    if explicit:
        targets = [explicit]
    elif idx_env:
        i = int(idx_env)
        if i < 0 or i >= len(all_dates):
            print(f"NG_INIT_INDEX={i} out of range (0..{len(all_dates) - 1}); "
                  f"nothing to do.", flush=True)
            return
        targets = [all_dates[i]]
    else:
        targets = all_dates
    print(f"year={YEAR} | {len(all_dates)} init-days available | "
          f"this process runs {len(targets)}: {targets[:3]}"
          f"{' ...' if len(targets) > 3 else ''}", flush=True)

    # Build the regridder once from a cheap metadata-only sample.
    sample = _rename(xarray.open_dataset(F_3D, chunks={})).isel(time=0)
    regridder = build_regridder(sample, model)

    for d in targets:
        process(model, regridder, ds3, dss, d)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
