#!/usr/bin/env python
"""Stage ERA5 DAILY statistics (min / max / 00 UTC value) for the heat-wave
definitions, one variable-year per process, from ARCO-ERA5.

Streams sub-daily fields and reduces to per-day statistics -- so the ECMWF-style
definitions (daily minima AND maxima above the 95th percentile) can be evaluated:

  * 2 m temperature (ECMWF): read HOURLY -> daily min (Tn) + max (Tx).
  * 850 hPa temperature (mixture + ours): read 6-HOURLY -> daily min + max + the
    00 UTC value (t00, for the single-value "ours" definition). T850's diurnal
    range is small, so 6-hourly captures min/max well.

Env:
  HW_YEAR       year (required).
  HW_ARCO_VAR   ARCO variable: '2m_temperature' | 'temperature'.
  HW_LEVEL      pressure level (only for 'temperature'; default 850).
  HW_CADENCE    hours between samples (1 = hourly, 6 = 6-hourly). Must divide 24.
  HW_TAG        output name tag: 't2m' | 't850'.
  HW_STATS      comma list of daily stats to write: any of min,max,val (val=00 UTC).
  HW_OUT_DIR    output dir (default .../era5_heatwave_daily).
  HW_DAY_BATCH  days streamed+reduced+written per batch (default 10).

Output: <out>/<tag>_daily_<year>.nc, dims (time=days, latitude, longitude), native
0.25 deg, variables <tag>_tmin / <tag>_tmax / <tag>_t00 (per HW_STATS). Resumable
(.part + _n_days_written; finished year skipped).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

from aic import config

# source zarr (public, anon). Default ARCO-ERA5 0.25 deg; override HW_ZARR with a
# WeatherBench2 coarse dataset (much cheaper) for the reference period.
SRC_ZARR = config.env_str(
    "HW_ZARR", "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3")

YEAR = int(config.env_required("HW_YEAR"))
ARCO_VAR = config.env_str("HW_ARCO_VAR", "temperature").strip()
LEVEL = config.env_int("HW_LEVEL", 850)
CADENCE = config.env_int("HW_CADENCE", 6)
TAG = config.env_str("HW_TAG", "t850").strip()
STATS = [s for s in config.env_list("HW_STATS", ["min", "max", "val"])]
DAY_BATCH = config.env_int("HW_DAY_BATCH", 10)
OUT_DIR = Path(config.env_str("HW_OUT_DIR", config.ERA5_HEATWAVE_DAILY))

if 24 % CADENCE != 0:
    raise SystemExit(f"HW_CADENCE must divide 24 (got {CADENCE})")
if any(s not in ("min", "max", "val") for s in STATS):
    raise SystemExit(f"HW_STATS must be from min,max,val (got {STATS})")

IS_SURFACE = (ARCO_VAR == "2m_temperature")
OUT_PATH = OUT_DIR / f"{TAG}_daily_{YEAR}.nc"
PART_PATH = OUT_PATH.with_suffix(".nc.part")


def day_index() -> pd.DatetimeIndex:
    # HW_END_DATE caps a partial/current year at the last day with real ERA5 data
    # (ARCO's time axis is pre-declared into the future but NaN beyond the front).
    end = (config.env_str("HW_END_DATE", "") or "").strip() or f"{YEAR}-12-31"
    return pd.date_range(f"{YEAR}-01-01", end, freq="1D")


def create_part(days, lat, lon) -> netCDF4.Dataset:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nc = netCDF4.Dataset(PART_PATH, "w", format="NETCDF4")
    nc.createDimension("time", len(days))
    nc.createDimension("latitude", len(lat))
    nc.createDimension("longitude", len(lon))
    units = "hours since 1900-01-01 00:00:00"
    vt = nc.createVariable("time", "f8", ("time",))
    vt.units = units; vt.calendar = "proleptic_gregorian"
    vt[:] = netCDF4.date2num(pd.to_datetime(days).to_pydatetime(), units,
                             "proleptic_gregorian")
    vlat = nc.createVariable("latitude", "f4", ("latitude",)); vlat[:] = lat
    vlat.units = "degrees_north"
    vlon = nc.createVariable("longitude", "f4", ("longitude",)); vlon[:] = lon
    vlon.units = "degrees_east"
    for s in STATS:
        v = nc.createVariable(f"{TAG}_t{s}", "f4", ("time", "latitude", "longitude"),
                              zlib=True, complevel=4, chunksizes=(1, len(lat), len(lon)))
        v.long_name = f"daily {s} of {ARCO_VAR}" + ("" if IS_SURFACE else f" at {LEVEL} hPa")
        v.units = "K"
    nc.source = (f"ERA5 (ARCO) daily {STATS} of {ARCO_VAR}"
                 f"{'' if IS_SURFACE else f' @ {LEVEL} hPa'}, {CADENCE}-hourly samples, "
                 f"year {YEAR}, native 0.25 deg. Heat-wave definition input.")
    nc.setncattr("_n_days_written", 0)
    nc.sync()
    return nc


def main() -> None:
    if OUT_PATH.exists():
        print(f"[daily] {OUT_PATH.name} complete -> skip", flush=True); return

    ds = xr.open_zarr(SRC_ZARR, chunks=None, storage_options=dict(token="anon"))
    da = ds[ARCO_VAR]
    if not IS_SURFACE:
        da = da.sel(level=LEVEL)
    tmax_front = pd.Timestamp(ds.time.values[-1])

    days = day_index()
    limit = config.env_int("HW_DAY_LIMIT", 0)            # smoke cap
    if limit > 0:
        days = days[:limit]
    nd = len(days)
    lat = ds.latitude.values
    lon = ds.longitude.values
    print(f"[daily] {YEAR} {TAG} ({ARCO_VAR}"
          f"{'' if IS_SURFACE else f'@{LEVEL}'}) cadence={CADENCE}h stats={STATS} "
          f"-> {nd} days, front {tmax_front.date()}", flush=True)

    if PART_PATH.exists():
        nc = netCDF4.Dataset(PART_PATH, "a")
        written = int(nc.getncattr("_n_days_written"))
        print(f"[daily] resuming {written}/{nd}", flush=True)
    else:
        nc = create_part(days, lat, lon)
        written = 0

    t0 = time.time()
    for d0 in range(written, nd, DAY_BATCH):
        d1 = min(d0 + DAY_BATCH, nd)
        batch = days[d0:d1]
        hrs = pd.date_range(batch[0], batch[-1] + pd.Timedelta(hours=24 - CADENCE),
                            freq=f"{CADENCE}h")
        hrs = hrs[hrs <= tmax_front]
        if len(hrs) == 0:
            break
        for attempt in range(6):
            try:
                sub = da.sel(time=hrs.values).compute()
                break
            except Exception as ex:
                w = 10 * (attempt + 1)
                print(f"  [retry {attempt}] days {d0}:{d1} {ex}; sleep {w}s", flush=True)
                time.sleep(w)
        else:
            raise RuntimeError(f"day batch {d0}:{d1} failed")
        grp = sub.groupby("time.dayofyear")
        n_got = len(np.unique(pd.to_datetime(sub.time.values).date))
        results = {}
        if "min" in STATS:
            results["min"] = grp.min().transpose("dayofyear", "latitude", "longitude").values
        if "max" in STATS:
            results["max"] = grp.max().transpose("dayofyear", "latitude", "longitude").values
        if "val" in STATS:
            results["val"] = sub.sel(time=[t for t in sub.time.values
                                           if pd.Timestamp(t).hour == 0]).transpose(
                "time", "latitude", "longitude").values
        for s in STATS:
            arr = results[s]
            nc.variables[f"{TAG}_t{s}"][d0:d0 + arr.shape[0], :, :] = arr.astype("float32")
        nc.setncattr("_n_days_written", d0 + n_got)
        nc.sync()
        rate = (d0 + n_got - written) / max(1e-9, time.time() - t0)
        print(f"  {d0 + n_got}/{nd} days  {rate:.2f} days/s", flush=True)
        if n_got < len(batch):
            break

    nc.close()
    PART_PATH.rename(OUT_PATH)
    print(f"[daily] DONE {OUT_PATH} ({OUT_PATH.stat().st_size/1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()
