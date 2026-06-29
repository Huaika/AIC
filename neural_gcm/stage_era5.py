#!/usr/bin/env python
"""Stage ONLY the ERA5 variables the NeuralGCM experiment consumes, one month per
process, into local storage -- so the GPU rollouts (and the downstream evaluation)
read locally and run deterministically (no per-task cloud streaming).

Variables staged (and nothing else):
  * INPUT  : the 7 model input variables on all 37 ERA5 levels
  * FORCING: sea_surface_temperature + sea_ice_cover

Cadence is **6-hourly**: the rollouts only init daily, but the downstream
evaluation (drift / spaghetti vs ERA5 truth) scores the 6 h prediction frames
against 6 h ERA5, so the 6 h cadence is required. This is exactly the variable
set + cadence the experiment needs -- no extra vars (w is dropped), no regridding
(ERA5 is already on the model's 0.25 deg grid), no level work (ERA5 ships the 37
levels the model wants). A pure SELECT at native 0.25 deg, mirroring the NextGEMS
input files.

Output: one NetCDF per (year, month) -> consumers open them with open_mfdataset.
Written incrementally with a ``_n_times_written`` attr; resumable. A finished month
is published by renaming ``.part`` -> ``.nc`` so consumers never see a half file.

Environment:
  ERA5_YEAR    year (required)
  ERA5_MONTH   month 1..12 (required)
  ERA5_OUT_DIR base output dir (default: ka_hc5935-ai-climate/era5_<year>/inputs)
  ERA5_BATCH   6 h steps streamed+written per batch (default 8)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

YEAR = int(os.environ["ERA5_YEAR"])
MONTH = int(os.environ["ERA5_MONTH"])
BATCH = int(os.environ.get("ERA5_BATCH", "8"))
OUT_DIR = Path(os.environ.get(
    "ERA5_OUT_DIR",
    f"/pfs/work9/workspace/scratch/ka_hc5935-ai-climate/era5_{YEAR}/inputs",
))

ERA5_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Exactly the model's variables (verified from the v1/deterministic_2_8_deg.pkl
# checkpoint). ERA5 long names == the names model.inputs_from_xarray expects, so
# no renaming is needed downstream.
INPUT_VARS = [
    "geopotential", "specific_humidity", "temperature",
    "u_component_of_wind", "v_component_of_wind",
    "specific_cloud_ice_water_content", "specific_cloud_liquid_water_content",
]
FORCING_VARS = ["sea_surface_temperature", "sea_ice_cover"]
ALL_VARS = INPUT_VARS + FORCING_VARS

OUT_PATH = OUT_DIR / f"era5_6hourly_{YEAR}_{MONTH:02d}.nc"
PART_PATH = OUT_PATH.with_suffix(".nc.part")


def month_times() -> np.ndarray:
    """6-hourly timestamps (00/06/12/18 Z) covering this calendar month.

    For a partial / current year, ERA5_END_DATE caps the series at the last day
    with real ERA5 data (the ARCO axis is pre-declared far into the future but
    only filled up to the near-real-time front). The cap is inclusive of the
    whole day, so e.g. ERA5_END_DATE=2026-06-23 keeps through 2026-06-23 18Z.
    A no-op for months that lie entirely before the cap.
    """
    start = pd.Timestamp(YEAR, MONTH, 1)
    end = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=18)
    times = pd.date_range(start, end, freq="6h").values.astype("datetime64[ns]")
    end_date = os.environ.get("ERA5_END_DATE", "").strip()
    if end_date:
        cutoff = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).to_datetime64()
        times = times[times < cutoff]
    limit = int(os.environ.get("ERA5_LIMIT", "0"))   # smoke-test cap
    return times[:limit] if limit > 0 else times


def create_part(times, lev, lat, lon) -> netCDF4.Dataset:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nc = netCDF4.Dataset(PART_PATH, "w", format="NETCDF4")
    nc.createDimension("time", len(times))
    nc.createDimension("level", len(lev))
    nc.createDimension("latitude", len(lat))
    nc.createDimension("longitude", len(lon))

    units = "hours since 1900-01-01 00:00:00"
    vt = nc.createVariable("time", "f8", ("time",))
    vt.units = units
    vt.calendar = "proleptic_gregorian"
    vt[:] = netCDF4.date2num(pd.to_datetime(times).to_pydatetime(), units,
                             "proleptic_gregorian")
    vlev = nc.createVariable("level", "i4", ("level",)); vlev[:] = lev
    vlev.units = "hPa"
    vlat = nc.createVariable("latitude", "f4", ("latitude",)); vlat[:] = lat
    vlat.units = "degrees_north"
    vlon = nc.createVariable("longitude", "f4", ("longitude",)); vlon[:] = lon
    vlon.units = "degrees_east"

    for v in INPUT_VARS:
        nc.createVariable(v, "f4", ("time", "level", "latitude", "longitude"),
                          zlib=False, chunksizes=(1, 1, len(lat), len(lon)))
    for v in FORCING_VARS:
        nc.createVariable(v, "f4", ("time", "latitude", "longitude"),
                          zlib=False, chunksizes=(1, len(lat), len(lon)))
    nc.source = (f"ERA5 (ARCO full_37) rollout inputs+forcing, year {YEAR} "
                 f"month {MONTH:02d}, 6-hourly, native 0.25 deg.")
    nc.setncattr("_n_times_written", 0)
    nc.sync()
    return nc


def main() -> None:
    if OUT_PATH.exists():
        print(f"[stage] {OUT_PATH.name} already complete -> skip", flush=True)
        return

    ds = xr.open_zarr(ERA5_PATH, chunks=None, storage_options=dict(token="anon"))
    times = month_times()
    nt = len(times)
    if nt == 0:
        print(f"[stage] {YEAR}-{MONTH:02d}: no times within ERA5_END_DATE cap "
              f"-> skip", flush=True)
        return
    lev = ds.level.values.astype("i4")
    lat = ds.latitude.values
    lon = ds.longitude.values
    print(f"[stage] {YEAR}-{MONTH:02d}: {nt} 6-hourly steps, {len(lev)} levels, "
          f"vars={ALL_VARS}", flush=True)

    if PART_PATH.exists():
        nc = netCDF4.Dataset(PART_PATH, "a")
        written = int(nc.getncattr("_n_times_written"))
        print(f"[stage] resuming part: {written}/{nt}", flush=True)
    else:
        nc = create_part(times, lev, lat, lon)
        written = 0

    t0 = time.time()
    for s in range(written, nt, BATCH):
        e = min(s + BATCH, nt)
        for attempt in range(6):
            try:
                sub = ds[ALL_VARS].sel(time=times[s:e]).compute()
                break
            except Exception as ex:
                w = 10 * (attempt + 1)
                print(f"  [retry {attempt}] {s}:{e} {ex}; sleep {w}s", flush=True)
                time.sleep(w)
        else:
            raise RuntimeError(f"batch {s}:{e} failed")
        for v in INPUT_VARS:
            nc.variables[v][s:e, :, :, :] = sub[v].astype("float32").values
        for v in FORCING_VARS:
            nc.variables[v][s:e, :, :] = sub[v].astype("float32").values
        nc.setncattr("_n_times_written", e)
        nc.sync()
        rate = (e - written) / max(1e-9, time.time() - t0)
        print(f"  {e}/{nt}  {rate:.2f} steps/s", flush=True)
    nc.close()

    PART_PATH.rename(OUT_PATH)
    print(f"[stage] DONE {OUT_PATH} ({OUT_PATH.stat().st_size / 1e9:.1f} GB)",
          flush=True)


if __name__ == "__main__":
    main()
