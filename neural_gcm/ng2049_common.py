#!/usr/bin/env python
"""Shared helpers for the NextGEMS-2049 diagnostic plots.

All three plot families (spaghetti, drift statistics, drift maps) are generated
for a configurable set of pressure levels between 0 and 1000 hPa. The level set
is driven by environment variables:

    NG_LEVEL_INTERVAL   spacing in hPa (default 50)  -> 50,100,...,1000
    NG_LEVEL_MIN        lowest level (default = interval)
    NG_LEVEL_MAX        highest level (default 1000)
    NG_LEVELS           explicit comma list, overrides the interval (e.g. "850,500")

Requested levels are intersected with the levels actually present in the
prediction NetCDFs (the model's 37 ERA5 levels); any that are not available are
reported and skipped (no silent truncation).

The expensive shared artefact is the NextGEMS-2049 temperature regridded onto the
128x64 model grid at all 25 native levels, for every 6 h step of the year
(``truth_modelgrid_T_native_2049.nc``, built once). Requested levels are obtained
by linear vertical interpolation of that cache -- the same 25->37 linear scheme
the rollouts used -- which is cheap and decouples the heavy read from the level
choice.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import gcsfs
import numpy as np
import xarray as xr

from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
import neuralgcm

YEAR = 2049
MODEL_NAME = "v1/deterministic_2_8_deg.pkl"
PRED_DIR = Path("/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/"
                "nextgems_2049/predictions")
DATA_3D = Path("/pfs/work9/workspace/scratch/ka_je2428-nextgems_2049/"
               "3D_nextgems_2049_6hourly_0.25deg_lat-lon.nc")
CONST_ZARR = Path("/pfs/work9/workspace/scratch/ka_je2428-nextgems_2049/"
                  "constant_fields.zarr")
OUTDIR = Path("results_daily_nextgems")
FIGROOT = Path("figures")
OUTDIR.mkdir(exist_ok=True)

NATIVE_TRUTH_NC = OUTDIR / f"truth_modelgrid_T_native_{YEAR}.nc"
TRUTH_BATCH = int(os.environ.get("NG_TRUTH_BATCH", "24"))   # time steps / regrid batch

RENAME_3D = {"t": "temperature", "lat": "latitude", "lon": "longitude"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def lat_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(da.latitude))
    return da.weighted(w).mean(["latitude", "longitude"])


def to_world(da: xr.DataArray) -> xr.DataArray:
    """Re-centre longitude to [-180,180) and order lat south->north (for maps)."""
    da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
    return da.sortby("longitude").sortby("latitude")


def figure_dir(kind: str) -> Path:
    d = FIGROOT / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


_COAST = None


def draw_coastlines(ax, lw: float = 0.4, color: str = "k", alpha: float = 0.7) -> None:
    """Overlay coastlines by contouring the NextGEMS land-sea mask at 0.5.

    Dependency-free (no cartopy): uses the 0.25 deg land_sea_mask from
    constant_fields.zarr, recentred to [-180,180) to match the map axes.
    """
    global _COAST
    if _COAST is None:
        cf = xr.open_zarr(CONST_ZARR)
        lsm = cf["land_sea_mask"]
        lsm = lsm.assign_coords(lon=(((lsm.lon + 180) % 360) - 180))
        lsm = lsm.sortby("lon").sortby("lat")
        _COAST = (lsm.lon.values, lsm.lat.values, lsm.values)
    lon, lat, mask = _COAST
    ax.contour(lon, lat, mask, levels=[0.5], colors=color,
               linewidths=lw, alpha=alpha)


# --------------------------------------------------------------------------- #
# level resolution
# --------------------------------------------------------------------------- #
_PRED_LEVELS = None
_PRED_GRID = None


def prediction_levels() -> list[int]:
    global _PRED_LEVELS
    if _PRED_LEVELS is None:
        f = sorted(PRED_DIR.glob(f"pred_{YEAR}_*.nc"))[0]
        with xr.open_dataset(f) as ds:
            _PRED_LEVELS = [int(x) for x in ds.level.values]
    return _PRED_LEVELS


def prediction_grid():
    """Canonical (latitude, longitude) coordinate values of the prediction grid."""
    global _PRED_GRID
    if _PRED_GRID is None:
        f = sorted(PRED_DIR.glob(f"pred_{YEAR}_*.nc"))[0]
        with xr.open_dataset(f) as ds:
            _PRED_GRID = (ds.latitude.values.copy(), ds.longitude.values.copy())
    return _PRED_GRID


def _interval() -> int:
    return int(os.environ.get("NG_LEVEL_INTERVAL", "50"))


def level_tag() -> str:
    """Short tag identifying the current level set, used in cache filenames."""
    if os.environ.get("NG_LEVELS", "").strip():
        return f"custom{len(requested_levels())}"
    return f"i{_interval()}"


def requested_levels() -> list[int]:
    explicit = os.environ.get("NG_LEVELS", "").strip()
    if explicit:
        req = [int(float(x)) for x in explicit.replace(",", " ").split()]
    else:
        interval = _interval()
        lo = int(os.environ.get("NG_LEVEL_MIN", str(interval)))
        hi = int(os.environ.get("NG_LEVEL_MAX", "1000"))
        req = list(range(lo, hi + 1, interval))
    req = [l for l in req if 0 < l <= 1000]
    avail = set(prediction_levels())
    kept = [l for l in req if l in avail]
    dropped = [l for l in req if l not in avail]
    if dropped:
        print(f"[levels] skipping {dropped} -- not in prediction grid "
              f"{sorted(avail)}")
    print(f"[levels] {len(kept)} levels: {kept}")
    return kept


# --------------------------------------------------------------------------- #
# NextGEMS truth on the model grid (native levels) + vertical interpolation
# --------------------------------------------------------------------------- #
def _build_regridder(sample: xr.DataArray):
    gcs = gcsfs.GCSFileSystem(token="anon")
    with gcs.open(f"gs://neuralgcm/models/{MODEL_NAME}", "rb") as f:
        model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(f))
    src = spherical_harmonic.Grid(
        latitude_nodes=sample.sizes["latitude"],
        longitude_nodes=sample.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(sample.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(sample.longitude),
    )
    return horizontal_interpolation.ConservativeRegridder(
        src, model.data_coords.horizontal, skipna=True)


def ensure_native_truth() -> None:
    """Build the all-native-levels model-grid NextGEMS T cache (once). Heavy I/O."""
    if NATIVE_TRUTH_NC.exists():
        print(f"[truth] cache present: {NATIVE_TRUTH_NC}")
        return
    print(f"[truth] building {NATIVE_TRUTH_NC} (all native levels, model grid) ...")
    ds = xr.open_dataset(DATA_3D, chunks={"time": TRUTH_BATCH}).rename(RENAME_3D)
    t = ds["temperature"]                       # (time, level=25, lat, lon)
    regridder = _build_regridder(t.isel(time=0))
    n = t.sizes["time"]
    slabs = []
    for s in range(0, n, TRUTH_BATCH):
        sub = t.isel(time=slice(s, s + TRUTH_BATCH)).compute()   # T has no NaN
        slabs.append(xarray_utils.regrid(sub, regridder))
        print(f"  {min(s + TRUTH_BATCH, n)}/{n}", flush=True)
    truth = xr.concat(slabs, dim="time")
    truth.name = "temperature"
    enc = {"temperature": {"dtype": "float32", "zlib": True, "complevel": 4}}
    truth.to_netcdf(NATIVE_TRUTH_NC, encoding=enc)
    print(f"[truth] wrote {NATIVE_TRUTH_NC} {dict(truth.sizes)}")


_TRUTH_AT = {}


def truth_at_levels(levels: list[int]) -> xr.DataArray:
    """NextGEMS T on the model grid, linearly vinterp'd to ``levels`` (cached)."""
    key = tuple(levels)
    if key in _TRUTH_AT:
        return _TRUTH_AT[key]
    ensure_native_truth()
    native = xr.open_dataarray(NATIVE_TRUTH_NC)
    out = native.interp(level=list(levels), method="linear",
                        kwargs={"fill_value": "extrapolate"}).load()
    native.close()
    # The regridded-truth grid and the prediction grid are the SAME model grid but
    # can differ by ~1e-14 due to separate float paths; xarray would then treat
    # them as distinct labels and outer-join (NaN gaps) on subtraction. Snap truth
    # onto the exact prediction coordinates so pred - truth stays on one grid.
    clat, clon = prediction_grid()
    out = out.reindex(latitude=clat, longitude=clon, method="nearest",
                      tolerance=1e-6)
    _TRUTH_AT[key] = out
    return out
