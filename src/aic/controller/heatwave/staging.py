#!/usr/bin/env python
"""Stage ERA5 850 hPa temperature (T850) for HEATWAVE analysis, 1991-2020.

The heatwave index (ECMWF/HWMId-style percentile method) is evaluated here on the
**850 hPa temperature** at a fixed sub-daily sampling cadence, over the 1991-2020
WMO normal period. From the staged T850 series the analysis computes, per calendar
day, the 90th-percentile threshold in a 31-day window pooled across the 30 years,
then flags spells of >= 3 consecutive days above it.

One YEAR per process (a Slurm array 0..29 maps to 1991..2020). Each task selects
the T850 timesteps at the requested cadence from ARCO-ERA5 and writes one NetCDF.

  HW_YEAR      year to stage (required).
  HW_LEVEL     pressure level in hPa (default 850).
  HW_CADENCE   hours between samples (default 24 -> one snapshot/day at HW_HOUR0;
               set 6 for 00/06/12/18 Z). Must divide 24.
  HW_HOUR0     first sample hour of the day in UTC (default 0 -> 00 UTC).
  HW_REGION    'world' (default) or any REGIONS key below (e.g. 'europe').
  HW_VAR       ARCO pressure-level variable (default 'temperature').
  HW_OUT_DIR   output dir (default .../ka_dm9435-ai-climate/era5_heatwave).
  HW_BATCH     timesteps streamed+written per batch (default 20; bounds memory).
  HW_LIMIT     cap on number of sample timesteps (0 = whole year; >0 for smoke).

COST NOTE: ARCO stores the 3-D temperature in 1-hour x all-37-levels x whole-globe
chunks ([1, 37, 721, 1440]), so selecting level 850 (or a region) does NOT reduce
the download -- the full 37-level global field is fetched per timestep and only the
stored output is a single level / cropped. The SAMPLING CADENCE is the only real
read-cost lever; 00 UTC-only (HW_CADENCE=24) is the minimum (~56 GB read/year).

Output: <out>/t850_<cadence>h_<region>_<year>.nc, dims (time, latitude, longitude)
with a scalar 'level' coord, native 0.25 deg, resumable via a .part file +
_n_times_written attr (published by renaming .part -> .nc). Consumers open the 30
years with xr.open_mfdataset for the threshold + spell analysis.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

ERA5_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Named regions: single source of truth.
from aic import config
from aic.regions import REGIONS, select_region

YEAR = int(config.env_required("HW_YEAR"))
LEVEL = config.env_int("HW_LEVEL", 850)
CADENCE = config.env_int("HW_CADENCE", 24)
HOUR0 = config.env_int("HW_HOUR0", 0)
REGION = config.env_str("HW_REGION", "world").strip().lower()
VAR = config.env_str("HW_VAR", "temperature").strip()
BATCH = config.env_int("HW_BATCH", 20)
OUT_DIR = Path(config.env_str("HW_OUT_DIR", config.ERA5_HEATWAVE))

if REGION not in REGIONS:
    raise SystemExit(f"HW_REGION must be one of {list(REGIONS)} (got {REGION!r})")
if 24 % CADENCE != 0:
    raise SystemExit(f"HW_CADENCE must divide 24 (got {CADENCE})")

OUT_PATH = OUT_DIR / f"t{LEVEL}_{CADENCE}h_{REGION}_{YEAR}.nc"
PART_PATH = OUT_PATH.with_suffix(".nc.part")


def crop_region(da: xr.DataArray) -> xr.DataArray:
    """Crop to this run's REGION box (no-op footprint for 'world')."""
    return select_region(da, REGION)


def sample_times(tmax: pd.Timestamp) -> np.ndarray:
    """Sampled timestamps for YEAR at HW_CADENCE starting at HW_HOUR0, capped at
    the ARCO data front tmax (a no-op for the historical 1991-2020 years)."""
    start = pd.Timestamp(YEAR, 1, 1) + pd.Timedelta(hours=HOUR0)
    end = pd.Timestamp(YEAR, 12, 31, 23)
    times = pd.date_range(start, end, freq=f"{CADENCE}h")
    times = times[times <= tmax]
    return times.values.astype("datetime64[ns]")


def create_part(times, lat, lon) -> netCDF4.Dataset:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nc = netCDF4.Dataset(PART_PATH, "w", format="NETCDF4")
    nc.createDimension("time", len(times))
    nc.createDimension("latitude", len(lat))
    nc.createDimension("longitude", len(lon))
    units = "hours since 1900-01-01 00:00:00"
    vt = nc.createVariable("time", "f8", ("time",))
    vt.units = units
    vt.calendar = "proleptic_gregorian"
    vt[:] = netCDF4.date2num(pd.to_datetime(times).to_pydatetime(), units,
                             "proleptic_gregorian")
    vlat = nc.createVariable("latitude", "f4", ("latitude",)); vlat[:] = lat
    vlat.units = "degrees_north"
    vlon = nc.createVariable("longitude", "f4", ("longitude",)); vlon[:] = lon
    vlon.units = "degrees_east"
    vlev = nc.createVariable("level", "i4"); vlev.assignValue(LEVEL)  # scalar coord
    vlev.units = "hPa"
    v = nc.createVariable("temperature", "f4", ("time", "latitude", "longitude"),
                          zlib=True, complevel=4,
                          chunksizes=(1, len(lat), len(lon)))
    v.long_name = f"{VAR} at {LEVEL} hPa"
    v.units = "K"
    v.coordinates = "level"
    nc.source = (f"ERA5 (ARCO full_37) {VAR} at {LEVEL} hPa, {CADENCE}-hourly "
                 f"(hour0={HOUR0} UTC), year {YEAR}, region {REGION}, native "
                 f"0.25 deg. Heatwave-analysis input (1991-2020 baseline).")
    nc.setncattr("_n_times_written", 0)
    nc.sync()
    return nc


def main() -> None:
    if OUT_PATH.exists():
        print(f"[hw] {OUT_PATH.name} already complete -> skip", flush=True)
        return

    ds = xr.open_zarr(ERA5_PATH, chunks=None, storage_options=dict(token="anon"))
    if VAR not in ds:
        raise SystemExit(f"{VAR!r} not in ARCO dataset")
    if LEVEL not in [int(x) for x in ds.level.values]:
        raise SystemExit(f"level {LEVEL} not in {sorted(int(x) for x in ds.level.values)}")

    tmax = pd.Timestamp(ds.time.values[-1])
    times = sample_times(tmax)
    limit = config.env_int("HW_LIMIT", 0)
    if limit > 0:
        times = times[:limit]
        print(f"[hw] HW_LIMIT={limit} -> smoke run, {len(times)} steps only",
              flush=True)
    nt = len(times)
    if nt == 0:
        print(f"[hw] {YEAR}: no sample times within ARCO front {tmax} -> skip",
              flush=True)
        return

    probe = crop_region(ds[VAR].isel(time=0).sel(level=LEVEL))
    rlat, rlon = probe.latitude.values, probe.longitude.values
    print(f"[hw] {YEAR} T{LEVEL} region={REGION} cadence={CADENCE}h hour0={HOUR0} "
          f"-> {nt} steps, out grid {len(rlat)}x{len(rlon)}", flush=True)

    if PART_PATH.exists():
        nc = netCDF4.Dataset(PART_PATH, "a")
        written = int(nc.getncattr("_n_times_written"))
        print(f"[hw] resuming part: {written}/{nt}", flush=True)
    else:
        nc = create_part(times, rlat, rlon)
        written = 0

    t0 = time.time()
    for s in range(written, nt, BATCH):
        e = min(s + BATCH, nt)
        for attempt in range(6):
            try:
                sub = ds[VAR].sel(time=times[s:e], level=LEVEL).compute()
                break
            except Exception as ex:
                w = 10 * (attempt + 1)
                print(f"  [retry {attempt}] {s}:{e} {ex}; sleep {w}s", flush=True)
                time.sleep(w)
        else:
            raise RuntimeError(f"batch {s}:{e} failed")
        sub = crop_region(sub).transpose("time", "latitude", "longitude")
        nc.variables["temperature"][s:e, :, :] = sub.astype("float32").values
        nc.setncattr("_n_times_written", e)
        nc.sync()
        rate = (e - written) / max(1e-9, time.time() - t0)
        print(f"  {e}/{nt}  {rate:.2f} steps/s", flush=True)
    nc.close()

    PART_PATH.rename(OUT_PATH)
    print(f"[hw] DONE {OUT_PATH} ({OUT_PATH.stat().st_size / 1e9:.2f} GB)",
          flush=True)


if __name__ == "__main__":
    main()
