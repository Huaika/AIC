#!/usr/bin/env python
"""Shared, RUN-AGNOSTIC helpers for the NeuralGCM rollout diagnostics.

ONE code path produces the spaghetti / drift-stat / drift-map plots for every
run -- the NextGEMS-2049 future-climate run and the ERA5-driven 1955 and 2023
runs -- selected purely by the ``EVAL_RUN`` environment variable:

    EVAL_RUN=nextgems2049   # reference = NextGEMS-2049 itself (no ERA5 truth)
    EVAL_RUN=era5_1955      # reference = ERA5 1955 (real truth)
    EVAL_RUN=era5_2023      # reference = ERA5 2023 (real truth)

The ONLY thing that differs between runs is where the prediction files live and
how the model-grid "truth" temperature is built. Both are encapsulated in the
``RUNS`` registry + ``truth_at_levels()`` below; the plot scripts are otherwise
identical for all three. (NextGEMS ships 25 native levels -> linear vinterp to
the requested levels; ERA5 already ships the 37 model levels -> the vinterp is a
no-op there. T has no NaNs, so a plain conservative regrid is used for both.)

The level set is configurable exactly as before (NG_LEVEL_INTERVAL etc.).
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

MODEL_NAME = "v1/deterministic_2_8_deg.pkl"
WS = "/pfs/work9/workspace/scratch"
_COAST_ZARR = f"{WS}/ka_je2428-nextgems_2049/constant_fields.zarr"  # land-sea mask

# --------------------------------------------------------------------------- #
# Run registry -- the only per-run configuration
# --------------------------------------------------------------------------- #
RUNS = {
    "nextgems2049": dict(
        year=2049,
        pred_dir=f"{WS}/ka_dm9435-ai-climate/nextgems_2049/predictions",
        ref_label="NextGEMS-2049",
        truth_kind="nextgems",
        truth_src=(f"{WS}/ka_je2428-nextgems_2049/"
                   "3D_nextgems_2049_6hourly_0.25deg_lat-lon.nc"),
        truth_rename={"t": "temperature", "lat": "latitude", "lon": "longitude"},
    ),
    "era5_1955": dict(
        year=1955,
        pred_dir=f"{WS}/ka_hc5935-ai-climate/era5_1955/predictions",
        ref_label="ERA5 1955",
        truth_kind="era5",
        truth_src=f"{WS}/ka_hc5935-ai-climate/era5_1955/inputs",
        truth_rename={},
    ),
    "era5_2023": dict(
        year=2023,
        pred_dir=f"{WS}/ka_hc5935-ai-climate/era5_2023/predictions",
        ref_label="ERA5 2023",
        truth_kind="era5",
        truth_src=f"{WS}/ka_hc5935-ai-climate/era5_2023/inputs",
        truth_rename={},
    ),
    # 2026 is a PARTIAL year: ERA5 staged Jan..Jun (capped at the 2026-06-23
    # data front), rollouts init 2026-01-01 .. 2026-06-13. Both pred + inputs
    # live in ka_dm9435's workspace (staged/rolled out here, not ka_hc5935's).
    "era5_2026": dict(
        year=2026,
        pred_dir=f"{WS}/ka_dm9435-ai-climate/era5_2026/predictions",
        ref_label="ERA5 2026",
        truth_kind="era5",
        truth_src=f"{WS}/ka_dm9435-ai-climate/era5_2026/inputs",
        truth_rename={},
    ),
}

RUN = os.environ.get("EVAL_RUN", "").strip()
if RUN not in RUNS:
    raise SystemExit(f"set EVAL_RUN to one of {list(RUNS)} (got {RUN!r})")
CFG = RUNS[RUN]
YEAR = CFG["year"]
REF_LABEL = CFG["ref_label"]
PRED_DIR = Path(CFG["pred_dir"])

OUTDIR = Path(f"results_eval_{RUN}")
FIGROOT = Path(f"figures/{RUN}")
OUTDIR.mkdir(exist_ok=True)

NATIVE_TRUTH_NC = OUTDIR / f"truth_modelgrid_T_{RUN}.nc"
TRUTH_BATCH = int(os.environ.get("EVAL_TRUTH_BATCH", "24"))


# --------------------------------------------------------------------------- #
# small helpers (unchanged, run-independent)
# --------------------------------------------------------------------------- #
def lat_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(da.latitude))
    return da.weighted(w).mean(["latitude", "longitude"])


def to_world(da: xr.DataArray) -> xr.DataArray:
    da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
    return da.sortby("longitude").sortby("latitude")


def figure_dir(kind: str) -> Path:
    d = FIGROOT / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


_COAST = None


def draw_coastlines(ax, lw: float = 0.4, color: str = "k", alpha: float = 0.7) -> None:
    """Coastlines by contouring the 0.25 deg land-sea mask at 0.5 (no cartopy).

    The mask is just a backdrop overlay (grid-independent), so the same NextGEMS
    constant-fields mask is reused for all runs.
    """
    global _COAST
    if _COAST is None:
        cf = xr.open_zarr(_COAST_ZARR)
        lsm = cf["land_sea_mask"]
        lsm = lsm.assign_coords(lon=(((lsm.lon + 180) % 360) - 180))
        lsm = lsm.sortby("lon").sortby("lat")
        _COAST = (lsm.lon.values, lsm.lat.values, lsm.values)
    lon, lat, mask = _COAST
    ax.contour(lon, lat, mask, levels=[0.5], colors=color, linewidths=lw, alpha=alpha)


# --------------------------------------------------------------------------- #
# level + grid resolution (from the prediction files; run-independent)
# --------------------------------------------------------------------------- #
_PRED_LEVELS = None
_PRED_GRID = None


def _first_pred() -> Path:
    fs = sorted(PRED_DIR.glob(f"pred_{YEAR}_*.nc"))
    if not fs:
        raise SystemExit(f"no prediction files in {PRED_DIR}")
    return fs[0]


def prediction_levels() -> list[int]:
    global _PRED_LEVELS
    if _PRED_LEVELS is None:
        with xr.open_dataset(_first_pred()) as ds:
            _PRED_LEVELS = [int(x) for x in ds.level.values]
    return _PRED_LEVELS


def prediction_grid():
    global _PRED_GRID
    if _PRED_GRID is None:
        with xr.open_dataset(_first_pred()) as ds:
            _PRED_GRID = (ds.latitude.values.copy(), ds.longitude.values.copy())
    return _PRED_GRID


def _interval() -> int:
    return int(os.environ.get("NG_LEVEL_INTERVAL", "50"))


def level_tag() -> str:
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
        print(f"[levels] skipping {dropped} -- not in prediction grid {sorted(avail)}")
    print(f"[levels] {len(kept)} levels: {kept}")
    return kept


# --------------------------------------------------------------------------- #
# Model-grid truth temperature (the only run-specific data path)
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


def _open_truth_source() -> xr.Dataset:
    """Open the run's reference dataset (NextGEMS file OR staged ERA5 months)."""
    if CFG["truth_kind"] == "nextgems":
        ds = xr.open_dataset(CFG["truth_src"], chunks={"time": TRUTH_BATCH})
    else:  # era5: the staged monthly files
        files = sorted(Path(CFG["truth_src"]).glob("era5_6hourly_*.nc"))
        if not files:
            raise SystemExit(f"no staged ERA5 files in {CFG['truth_src']}")
        ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": TRUTH_BATCH})
    if CFG["truth_rename"]:
        ds = ds.rename(CFG["truth_rename"])
    return ds


def ensure_native_truth() -> None:
    """Build the all-native-levels model-grid reference-T cache (once). Heavy I/O."""
    if NATIVE_TRUTH_NC.exists():
        print(f"[truth] cache present: {NATIVE_TRUTH_NC}")
        return
    print(f"[truth] building {NATIVE_TRUTH_NC} ({REF_LABEL}, model grid) ...")
    ds = _open_truth_source()
    t = ds["temperature"]                       # (time, level, lat, lon)
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
    """Reference T on the model grid, linearly vinterp'd to ``levels`` (cached).

    For ERA5 the native levels already include the requested ones, so the linear
    interpolation is effectively a select; for NextGEMS it interpolates 25->levels.
    """
    key = tuple(levels)
    if key in _TRUTH_AT:
        return _TRUTH_AT[key]
    ensure_native_truth()
    native = xr.open_dataarray(NATIVE_TRUTH_NC)
    out = native.interp(level=list(levels), method="linear",
                        kwargs={"fill_value": "extrapolate"}).load()
    native.close()
    clat, clon = prediction_grid()
    out = out.reindex(latitude=clat, longitude=clon, method="nearest", tolerance=1e-6)
    _TRUTH_AT[key] = out
    return out
