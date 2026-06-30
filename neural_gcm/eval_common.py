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
no-op there. The 3D prognostics have no NaNs, so a plain conservative regrid is
used for both.)

Every plot family runs for each variable in ``selected_variables()`` -- the five
core prognostics by default (temperature, geopotential, specific humidity, u/v
wind), overridable via EVAL_VARS (see VARIABLES for the full plottable set, which
also includes the two cloud-water-content fields). The level set is configurable
exactly as before (NG_LEVEL_INTERVAL etc.).
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
    ),
    "era5_1955": dict(
        year=1955,
        pred_dir=f"{WS}/ka_hc5935-ai-climate/era5_1955/predictions",
        ref_label="ERA5 1955",
        truth_kind="era5",
        truth_src=f"{WS}/ka_hc5935-ai-climate/era5_1955/inputs",
    ),
    "era5_2023": dict(
        year=2023,
        pred_dir=f"{WS}/ka_hc5935-ai-climate/era5_2023/predictions",
        ref_label="ERA5 2023",
        truth_kind="era5",
        truth_src=f"{WS}/ka_hc5935-ai-climate/era5_2023/inputs",
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

TRUTH_BATCH = int(os.environ.get("EVAL_TRUTH_BATCH", "24"))

# --------------------------------------------------------------------------- #
# Variable registry -- the prognostic 3D fields present in every prediction
# file (and in both truth sources). ``short`` is the filename/label tag, used
# for cache + figure names; ``nextgems_src`` is that variable's short name in
# the NextGEMS 3D file (ERA5 inputs already use the canonical NeuralGCM names).
# ``cmap`` is the field colormap for drift maps (drift itself is always RdBu_r).
# --------------------------------------------------------------------------- #
VARIABLES = {
    "temperature":         dict(short="T", units="K",       nextgems_src="t",
                                 cmap="RdYlBu_r", label="temperature"),
    "geopotential":        dict(short="Z", units="m^2/s^2", nextgems_src="z",
                                 cmap="viridis",  label="geopotential"),
    "specific_humidity":   dict(short="Q", units="kg/kg",   nextgems_src="q",
                                 cmap="viridis",  label="specific humidity"),
    "u_component_of_wind": dict(short="U", units="m/s",      nextgems_src="u",
                                 cmap="RdBu_r",   label="u-wind"),
    "v_component_of_wind": dict(short="V", units="m/s",      nextgems_src="v",
                                 cmap="RdBu_r",   label="v-wind"),
    "specific_cloud_ice_water_content":    dict(short="CIWC", units="kg/kg",
                                 nextgems_src="ciwc", cmap="viridis",
                                 label="cloud ice water content"),
    "specific_cloud_liquid_water_content": dict(short="CLWC", units="kg/kg",
                                 nextgems_src="clwc", cmap="viridis",
                                 label="cloud liquid water content"),
}

# The five core prognostics plotted by default; override with EVAL_VARS
# (comma/space list of canonical names, e.g. EVAL_VARS="temperature,geopotential").
DEFAULT_VARS = ["temperature", "geopotential", "specific_humidity",
                "u_component_of_wind", "v_component_of_wind"]


def selected_variables() -> list[str]:
    env = os.environ.get("EVAL_VARS", "").strip()
    vs = ([v.strip() for v in env.replace(",", " ").split()] if env
          else list(DEFAULT_VARS))
    bad = [v for v in vs if v not in VARIABLES]
    if bad:
        raise SystemExit(f"unknown EVAL_VARS {bad}; choose from {list(VARIABLES)}")
    print(f"[vars] {len(vs)} variable(s): {vs}")
    return vs


def native_truth_nc(var: str) -> Path:
    """Per-variable all-native-levels model-grid truth cache path.

    temperature -> truth_modelgrid_T_<run>.nc (matches the pre-existing T cache,
    so it is reused rather than rebuilt).
    """
    return OUTDIR / f"truth_modelgrid_{VARIABLES[var]['short']}_{RUN}.nc"


# Per-run directory holding the per-time-chunk partial truth files produced by
# the chunked (30-min-job) build; finalize_truth() concatenates them.
PARTS_DIR = OUTDIR / "truth_parts" / RUN


def _part_path(var: str, t0: int, t1: int) -> Path:
    return PARTS_DIR / f"{VARIABLES[var]['short']}_{t0:05d}_{t1:05d}.nc"


# --------------------------------------------------------------------------- #
# small helpers (unchanged, run-independent)
# --------------------------------------------------------------------------- #
def lat_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(da.latitude))
    return da.weighted(w).mean(["latitude", "longitude"])


def to_world(da: xr.DataArray) -> xr.DataArray:
    da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
    return da.sortby("longitude").sortby("latitude")


def figure_dir(variable: str, kind: str) -> Path:
    """figures/<run>/<full-variable-name>/<kind>/  (kind = spaghetti|drift_stats|
    drift_maps). The variable is its own folder -- the short tag is NOT in the
    filename anymore -- so one variable's diagnostics live together."""
    d = FIGROOT / variable / kind
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


def _truth_rename(vars: list[str]) -> dict:
    """Source->canonical rename. NextGEMS uses short var names + lat/lon; the
    staged ERA5 files already use the canonical NeuralGCM names."""
    if CFG["truth_kind"] != "nextgems":
        return {}
    m = {VARIABLES[v]["nextgems_src"]: v for v in vars}
    m.update({"lat": "latitude", "lon": "longitude"})
    return m


def _open_truth_source() -> xr.Dataset:
    """Open the run's reference dataset (NextGEMS file OR staged ERA5 months)."""
    if CFG["truth_kind"] == "nextgems":
        return xr.open_dataset(CFG["truth_src"], chunks={"time": TRUTH_BATCH})
    files = sorted(Path(CFG["truth_src"]).glob("era5_6hourly_*.nc"))
    if not files:
        raise SystemExit(f"no staged ERA5 files in {CFG['truth_src']}")
    return xr.open_mfdataset(files, combine="by_coords", chunks={"time": TRUTH_BATCH})


def ensure_native_truth(vars: list[str] | None = None) -> None:
    """Build the all-native-levels model-grid truth cache(s). Heavy I/O.

    Builds ONE cache file per variable, but does so in a SINGLE pass over the
    (large) source: open once, regrid every still-missing variable per time
    slab. Call this once (with the full variable set) before the plot scripts;
    ``truth_at_levels`` then just opens the caches. The 3D prognostic fields are
    clean (no NaN, physical ranges), so a plain conservative regrid is used --
    only the surface sst/ci forcing needs masking, and that is not plotted here.
    """
    vars = vars or selected_variables()
    missing = [v for v in vars if not native_truth_nc(v).exists()]
    present = [v for v in vars if v not in missing]
    if present:
        print(f"[truth] caches present: {present}")
    if not missing:
        return
    print(f"[truth] building {missing} ({REF_LABEL}, model grid, single pass) ...")
    ds = _open_truth_source()
    rename = _truth_rename(missing)
    if rename:
        ds = ds.rename(rename)
    regridder = _build_regridder(ds[missing[0]].isel(time=0))
    n = ds.sizes["time"]
    slabs = {v: [] for v in missing}
    for s in range(0, n, TRUTH_BATCH):
        sub = ds[missing].isel(time=slice(s, s + TRUTH_BATCH)).compute()
        for v in missing:
            slabs[v].append(xarray_utils.regrid(sub[v], regridder))
        print(f"  {min(s + TRUTH_BATCH, n)}/{n}", flush=True)
    for v in missing:
        truth = xr.concat(slabs[v], dim="time")
        truth.name = v
        enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4}}
        truth.to_netcdf(native_truth_nc(v), encoding=enc)
        print(f"[truth] wrote {native_truth_nc(v)} {dict(truth.sizes)}")


_TRUTH_AT = {}


def truth_at_levels(var: str, levels: list[int]) -> xr.DataArray:
    """Reference ``var`` on the model grid, linearly vinterp'd to ``levels``.

    For ERA5 the native levels already include the requested ones, so the linear
    interpolation is effectively a select; for NextGEMS it interpolates 25->levels.
    """
    key = (var, tuple(levels))
    if key in _TRUTH_AT:
        return _TRUTH_AT[key]
    if not native_truth_nc(var).exists() and os.environ.get("EVAL_REQUIRE_CACHE"):
        raise SystemExit(
            f"[truth] cache missing for {var} ({native_truth_nc(var)}) and "
            f"EVAL_REQUIRE_CACHE is set -- build it first via the truth-chunk + "
            f"finalize jobs. Refusing to build a multi-hour cache inside this job.")
    ensure_native_truth([var])
    native = xr.open_dataarray(native_truth_nc(var))
    out = native.interp(level=list(levels), method="linear",
                        kwargs={"fill_value": "extrapolate"}).load()
    native.close()
    clat, clon = prediction_grid()
    out = out.reindex(latitude=clat, longitude=clon, method="nearest", tolerance=1e-6)
    _TRUTH_AT[key] = out
    return out


# --------------------------------------------------------------------------- #
# Chunked, resumable truth build -- so the (otherwise multi-hour) NextGEMS/ERA5
# truth caches fit inside the 30-min dev_cpu lane. build_truth_chunk() regrids
# one time-slice of every still-missing variable and writes a per-chunk part
# file; finalize_truth() concatenates the parts into the final cache once they
# are all present. Both are resumable: existing parts/caches are skipped.
# --------------------------------------------------------------------------- #
def truth_source_nsteps() -> int:
    ds = _open_truth_source()
    n = int(ds.sizes["time"])
    ds.close()
    return n


def build_truth_chunk(t0: int, t1: int, vars: list[str] | None = None) -> None:
    """Regrid timesteps [t0, t1) of every missing variable -> per-chunk parts.

    A variable is skipped if its final cache OR this chunk's part already exists.
    Reads the source in TRUTH_BATCH sub-slabs to bound memory; writes each part
    atomically (.tmp -> rename)."""
    vars = vars or selected_variables()
    want = [v for v in vars
            if not native_truth_nc(v).exists() and not _part_path(v, t0, t1).exists()]
    if not want:
        print(f"[chunk {t0}:{t1}] nothing to do (caches/parts present)")
        return
    ds = _open_truth_source()
    n = int(ds.sizes["time"])
    if t0 >= n:
        print(f"[chunk {t0}:{t1}] beyond data (n={n}); no-op")
        ds.close()
        return
    t1 = min(t1, n)
    rename = _truth_rename(want)
    if rename:
        ds = ds.rename(rename)
    print(f"[chunk {t0}:{t1}] building {want} ({REF_LABEL}); n={n}", flush=True)
    regridder = _build_regridder(ds[want[0]].isel(time=0))
    slabs = {v: [] for v in want}
    for s in range(t0, t1, TRUTH_BATCH):
        e = min(s + TRUTH_BATCH, t1)
        sub = ds[want].isel(time=slice(s, e)).compute()
        for v in want:
            slabs[v].append(xarray_utils.regrid(sub[v], regridder))
        print(f"  {e - t0}/{t1 - t0} (abs {e}/{n})", flush=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    for v in want:
        part = xr.concat(slabs[v], dim="time")
        part.name = v
        enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4}}
        p = _part_path(v, t0, t1)
        tmp = p.with_suffix(".tmp.nc")
        part.to_netcdf(tmp, encoding=enc)
        tmp.rename(p)
        print(f"[chunk] wrote {p} {dict(part.sizes)}", flush=True)


def finalize_truth(vars: list[str] | None = None) -> None:
    """Concatenate per-chunk parts into the final per-variable cache.

    Only finalizes a variable whose parts together cover ALL timesteps (so an
    incomplete build is left for a re-run rather than producing a short cache).
    Deletes the parts after a successful write."""
    vars = vars or selected_variables()
    n = None
    for v in vars:
        fc = native_truth_nc(v)
        if fc.exists():
            print(f"[final] {v}: cache present")
            continue
        parts = sorted(PARTS_DIR.glob(f"{VARIABLES[v]['short']}_*.nc"))
        if not parts:
            print(f"[final] {v}: no parts found in {PARTS_DIR}")
            continue
        das = [xr.open_dataarray(p) for p in parts]
        full = xr.concat(das, dim="time").sortby("time")
        full = full.drop_duplicates("time")
        if n is None:
            n = truth_source_nsteps()
        if int(full.sizes["time"]) != n:
            print(f"[final] {v}: INCOMPLETE {int(full.sizes['time'])}/{n} steps "
                  f"({len(parts)} parts) -- skip, re-run chunks")
            for d in das:
                d.close()
            continue
        full.name = v
        enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4}}
        tmp = fc.with_suffix(".tmp.nc")
        full.to_netcdf(tmp, encoding=enc)
        tmp.rename(fc)
        for d in das:
            d.close()
        for p in parts:
            p.unlink()
        print(f"[final] wrote {fc} {dict(full.sizes)} (from {len(parts)} parts, removed)")
