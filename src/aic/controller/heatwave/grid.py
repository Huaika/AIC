#!/usr/bin/env python
"""Shared regridding onto NeuralGCM's 2.8 deg model grid, used by every heat-wave
analysis (the daily-stats definitions here, and the view animations). One place
builds the conservative regridder + caches regridded fields, so no analysis re-pays
the regrid and the whole project stays on the same grid.

The source grid is INFERRED from each input (equiangular ARCO 0.25 deg or a
WeatherBench2 coarse grid), so mixed-source inputs (WB2 reference + ARCO 2023) both
land on the identical model grid.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import xarray as xr
import gcsfs
import neuralgcm
from dinosaur import spherical_harmonic, xarray_utils, horizontal_interpolation

MODEL_NAME = "v1/deterministic_2_8_deg.pkl"
_MODEL_HGRID = None


def model_hgrid():
    """NeuralGCM's horizontal grid (128x64, 2.8 deg), loaded once."""
    global _MODEL_HGRID
    if _MODEL_HGRID is None:
        gcs = gcsfs.GCSFileSystem(token="anon")
        with gcs.open(f"gs://neuralgcm/models/{MODEL_NAME}", "rb") as f:
            _MODEL_HGRID = neuralgcm.PressureLevelModel.from_checkpoint(
                pickle.load(f)).data_coords.horizontal
    return _MODEL_HGRID


def build_regridder(sample):
    """Conservative regridder from `sample`'s native grid to the model grid."""
    src = spherical_harmonic.Grid(
        latitude_nodes=sample.sizes["latitude"],
        longitude_nodes=sample.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(sample.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(sample.longitude))
    return horizontal_interpolation.ConservativeRegridder(
        src, model_hgrid(), skipna=True)


def regrid_da(da, regridder=None, batch=None):
    """Regrid a (time, latitude, longitude) DataArray to the model grid. Pass
    `batch` to regrid in time-slabs (bounds memory for large 0.25 deg inputs)."""
    if regridder is None:
        regridder = build_regridder(da.isel(time=0))
    if batch is None:
        out = xarray_utils.regrid(da, regridder)
    else:
        import xarray as _xr
        n = da.sizes["time"]
        out = _xr.concat([xarray_utils.regrid(
            da.isel(time=slice(s, s + batch)).compute(), regridder)
            for s in range(0, n, batch)], dim="time")
    return out.transpose("time", "latitude", "longitude")


def load_daily_regridded(daily_dir, tag, year, stats, cache_dir):
    """Open <tag>_daily_<year>.nc, regrid the requested `stats` onto the model grid,
    and cache to <cache_dir>/<tag>_daily_<year>_2p8deg.nc. Returns an xr.Dataset with
    variables '<tag>_<stat>' (time, latitude, longitude) -- so ref (WB2) and target
    (ARCO) years, on different native grids, come back on the identical model grid."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{tag}_daily_{year}_2p8deg.nc"
    if cache.exists():
        return xr.open_dataset(cache)
    src = xr.open_dataset(Path(daily_dir) / f"{tag}_daily_{year}.nc")
    reg = build_regridder(src[f"{tag}_{stats[0]}"].isel(time=0))
    out = {f"{tag}_{s}": regrid_da(src[f"{tag}_{s}"], reg) for s in stats}
    res = xr.Dataset(out)
    res.to_netcdf(cache, encoding={v: {"dtype": "float32", "zlib": True,
                                       "complevel": 4} for v in res.data_vars})
    src.close()
    print(f"[grid] regridded {tag} {year} -> {cache.name}", flush=True)
    return res
