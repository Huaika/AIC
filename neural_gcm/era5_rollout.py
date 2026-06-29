#!/usr/bin/env python
"""NeuralGCM daily 10-day/6 h rollouts driven by ERA5, for years 1955 and 2023.

ERA5 analogue of ``nextgems_2049_rollout.py``. Differences from the NextGEMS
version (because ERA5 is cleaner and already on the model's coordinates):
  * Source = the LOCAL pre-staged ERA5 monthly files in
    ``ka_hc5935-ai-climate/era5_<year>/inputs/era5_6hourly_<year>_<MM>.nc``
    (opened together with open_mfdataset) -- no cloud streaming at run time.
  * Variables already carry the model's names -> no rename.
  * ERA5 already ships the 37 model levels -> NO 25->37 vertical interpolation.
  * ERA5 sst/sea_ice are physically clean (NaN only over land, a static mask) ->
    no per-timestep junk-masking; a single regrid + fill_nan_with_nearest suffices.
  * Forcing uses ERA5's 24 h ``selective_temporal_shift`` (the convention from the
    validated ERA5 path ``rackow_daily_rollouts.ipynb``).
Unlike NextGEMS-2049, ERA5 IS the truth, so the downstream evaluation can score
these rollouts against ERA5 directly.

One run == one init-day == one 10-day/6 h forecast, written as one NetCDF with the
full prediction fields (identical schema to pred_2049_<date>.nc). Resumable: an
init-day whose output already exists is skipped.

To minimise Slurm overhead this process handles a BATCH of init-days (loads the
model + builds the regridder once, then loops): see ERA5_INIT_START / _COUNT.

Environment:
  ERA5_YEAR          year (1955 or 2023)
  ERA5_INIT_START    0-based index into the year's init-day list (batch start)
  ERA5_INIT_COUNT    number of consecutive init-days this process runs (default 10)
  ERA5_IN_DIR        staged inputs dir (default ka_hc5935-ai-climate/era5_<year>/inputs)
  ERA5_OUT_DIR       predictions dir (default ka_hc5935-ai-climate/era5_<year>/predictions)
  ERA5_MODEL         checkpoint (default v1/deterministic_2_8_deg.pkl)
  ERA5_ROLLOUT_DAYS  rollout length (default 10)
  ERA5_OUT_H         output cadence h (default 6)
  ERA5_SST_STRIDE_H  forcing sampling stride h (default 24)
  ERA5_INIT_STRIDE_DAYS  spacing between init-days (default 1)
  ERA5_SEED          PRNG seed for encode (default 42)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

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
YEAR = int(os.environ["ERA5_YEAR"])
IN_DIR = Path(os.environ.get(
    "ERA5_IN_DIR",
    f"/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_{YEAR}/inputs"))
OUT_DIR = Path(os.environ.get(
    "ERA5_OUT_DIR",
    f"/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_{YEAR}/predictions"))
MODEL_NAME = os.environ.get("ERA5_MODEL", "v1/deterministic_2_8_deg.pkl")
ROLLOUT_DAYS = int(os.environ.get("ERA5_ROLLOUT_DAYS", "10"))
OUT_H = int(os.environ.get("ERA5_OUT_H", "6"))
SST_STRIDE_H = int(os.environ.get("ERA5_SST_STRIDE_H", "24"))
INIT_STRIDE_DAYS = int(os.environ.get("ERA5_INIT_STRIDE_DAYS", "1"))
SEED = int(os.environ.get("ERA5_SEED", "42"))


def load_model() -> neuralgcm.PressureLevelModel:
    gcs = gcsfs.GCSFileSystem(token="anon")
    with gcs.open(f"gs://neuralgcm/models/{MODEL_NAME}", "rb") as f:
        ckpt = pickle.load(f)
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)


def open_era5() -> xarray.Dataset:
    files = sorted(IN_DIR.glob(f"era5_6hourly_{YEAR}_*.nc"))
    if not files:
        raise SystemExit(f"no staged ERA5 files in {IN_DIR}")
    return xarray.open_mfdataset(files, combine="by_coords")


def build_regridder(sample, model):
    src = spherical_harmonic.Grid(
        latitude_nodes=sample.sizes["latitude"],
        longitude_nodes=sample.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(sample.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(sample.longitude),
    )
    return horizontal_interpolation.ConservativeRegridder(
        src, model.data_coords.horizontal, skipna=True)


def _end_str(init_date: str, days: int) -> str:
    return str(np.datetime64(init_date) + np.timedelta64(days, "D"))


# --------------------------------------------------------------------------- #
# Per-init-day forecast
# --------------------------------------------------------------------------- #
def run_one(model, regridder, ds, init_date: str) -> xarray.Dataset:
    def regrid(d):
        return xarray_utils.fill_nan_with_nearest(xarray_utils.regrid(d, regridder))

    # --- atmospheric inputs at the single initial time t0 (already 37 levels) ---
    t0 = ds[model.input_variables].sel(time=init_date, method="nearest").compute()
    if "time" not in t0.dims:
        t0 = t0.expand_dims(time=[t0.time.values])
    t0_rg = regrid(t0)

    # --- prescribed time-varying SST / sea-ice forcing over the rollout ---
    # ERA5 convention: 24 h selective temporal shift (matches rackow ERA5 path).
    forcing = (
        ds[model.forcing_variables]
        .pipe(xarray_utils.selective_temporal_shift,
              variables=model.forcing_variables, time_shift="24 hours")
        .sel(time=slice(init_date, _end_str(init_date, ROLLOUT_DAYS), SST_STRIDE_H))
        .compute()
    )
    forcing_rg = regrid(forcing)   # ERA5 land mask is static -> single regrid ok

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
        steps=steps, timedelta=timedelta, start_with_input=True)

    pred = model.data_to_xarray(predictions, times=lead_h)
    init_dt = np.datetime64(init_date)
    valid_time = init_dt + (lead_h * np.timedelta64(1, "h"))
    pred = pred.assign_coords(
        lead_hours=("time", lead_h.astype(np.int32)),
        valid_time=("time", valid_time))
    pred.attrs.update(
        source=f"NeuralGCM rollout forced by ERA5 {YEAR}",
        model=MODEL_NAME, init_date=str(np.datetime64(init_date, "D")),
        rollout_days=ROLLOUT_DAYS, output_hours=OUT_H,
        sst_stride_hours=SST_STRIDE_H, seed=SEED)
    return pred


# --------------------------------------------------------------------------- #
# Init-day selection + batch driver
# --------------------------------------------------------------------------- #
def init_dates_for_year(ds) -> list[str]:
    last = pd.Timestamp(np.datetime64(ds.time.values[-1]))
    latest_init = last - pd.Timedelta(days=ROLLOUT_DAYS)
    dates = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31",
                          freq=f"{INIT_STRIDE_DAYS}D")
    return [str(d.date()) for d in dates if d <= latest_init]


def out_path(init_date: str) -> Path:
    return OUT_DIR / f"pred_{YEAR}_{init_date}.nc"


def process(model, regridder, ds, init_date: str) -> None:
    out = out_path(init_date)
    if out.exists():
        print(f"[{init_date}] already done -> skip", flush=True)
        return
    print(f"[{init_date}] running {ROLLOUT_DAYS}-day/{OUT_H}h rollout ...", flush=True)
    t0 = time.time()
    pred = run_one(model, regridder, ds, init_date)
    tmp = out.with_suffix(".nc.tmp")
    enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4}
           for v in pred.data_vars}
    pred.to_netcdf(tmp, encoding=enc)
    tmp.rename(out)
    dt = time.time() - t0
    print(f"[{init_date}] wrote {out.name} "
          f"({out.stat().st_size / 1e6:.1f} MB) in {dt:.1f}s "
          f"({dt / 60:.2f} min/day)", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("JAX devices:", jax.devices(), flush=True)

    ds = open_era5()
    all_dates = init_dates_for_year(ds)

    start = int(os.environ.get("ERA5_INIT_START", "0"))
    count = int(os.environ.get("ERA5_INIT_COUNT", "10"))
    targets = all_dates[start:start + count]
    if not targets:
        print(f"ERA5_INIT_START={start} beyond {len(all_dates)} init-days; "
              f"nothing to do.", flush=True)
        return
    print(f"year={YEAR} | {len(all_dates)} init-days total | this process runs "
          f"[{start}:{start + count}] = {len(targets)}: {targets}", flush=True)

    model = load_model()
    regridder = build_regridder(ds.isel(time=0), model)

    for d in targets:
        process(model, regridder, ds, d)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
