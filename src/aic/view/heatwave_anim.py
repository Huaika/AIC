#!/usr/bin/env python
"""Animate detected heat waves on a regional map (GIF), coloured by the day's
standardized T850 anomaly, for the best-performing reference window (+/-5 days).

Pipeline:
  1. Build (or load) a cached day-of-year CLIMATOLOGY on the 2.8 deg grid from
     1991-2020: per calendar day (pseudo-year, 1..366) and grid cell, the 95th-
     percentile heat-wave THRESHOLD, the MEAN and the STD over a +/-window window.
     Saved once to clim_w<window>_2p8deg.nc -> future animations skip the 30-yr
     regrid entirely (this is the expensive step).
  2. Regrid the target year (2023) to the 2.8 deg grid, crop to the region.
  3. Per cell: hot = T850 > threshold[doy]; z = (T850 - mean[doy]) / std[doy];
     a cell is "affected" on days inside a >= 3-consecutive-hot-day spell.
  4. GIF: one 1-second frame per day on which any region cell is affected. The
     WHOLE anomaly field is drawn faint (30% opacity); the currently-affected
     cells are overdrawn at full opacity so only the active heat wave stands out.

Env: HW_WINDOW (default 5), HW_SPEC_REGION (default europe), HW_YEAR (default 2023).
Only the region's cells drive the frame list. Coloured by (T-mean)/std (RdBu_r).
"""
from __future__ import annotations
import glob
import io
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs
import neuralgcm
from dinosaur import spherical_harmonic, xarray_utils, horizontal_interpolation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

from aic.regions import REGIONS, region_mask, region_extent

DATA = os.environ.get(
    "HW_DATA_DIR", "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave")
CLIM_DIR = Path(os.environ.get(
    "HW_CLIM_DIR", "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_clim"))
GIF_DIR = Path(os.environ.get(
    "HW_GIF_DIR", "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_gifs"))
WINDOW = int(os.environ.get("HW_WINDOW", "5"))       # best-performing window
REGION = os.environ.get("HW_SPEC_REGION", "europe").strip().lower()
YEAR = int(os.environ.get("HW_YEAR", "2023"))
MIN_DUR = int(os.environ.get("HW_MIN_DURATION", "3"))
Q = float(os.environ.get("HW_Q", "0.95"))
MODEL_NAME = "v1/deterministic_2_8_deg.pkl"
REGRID_BATCH = int(os.environ.get("HW_REGRID_BATCH", "300"))
ZLIM = float(os.environ.get("HW_ZLIM", "4.0"))       # colour scale +/- sigma
_COAST_ZARR = "/pfs/work9/workspace/scratch/ka_je2428-nextgems_2049/constant_fields.zarr"


def _year(f):
    return int(os.path.basename(f).split("_")[-1].split(".")[0])


def build_regridder(sample):
    gcs = gcsfs.GCSFileSystem(token="anon")
    with gcs.open(f"gs://neuralgcm/models/{MODEL_NAME}", "rb") as f:
        model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(f))
    src = spherical_harmonic.Grid(
        latitude_nodes=sample.sizes["latitude"],
        longitude_nodes=sample.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(sample.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(sample.longitude))
    return horizontal_interpolation.ConservativeRegridder(
        src, model.data_coords.horizontal, skipna=True)


def load_regridded(files, regridder):
    ds = xr.open_mfdataset(sorted(files), combine="by_coords")
    da = ds["temperature"]
    n = da.sizes["time"]
    chunks = []
    for s in range(0, n, REGRID_BATCH):
        chunks.append(xarray_utils.regrid(
            da.isel(time=slice(s, s + REGRID_BATCH)).compute(), regridder))
        print(f"    regridded {min(s+REGRID_BATCH, n)}/{n}", flush=True)
    reg = xr.concat(chunks, dim="time").transpose("time", "latitude", "longitude")
    times = pd.to_datetime(reg["time"].values)
    return (reg.values.astype("float32"), times,
            reg["latitude"].values, reg["longitude"].values)


def build_or_load_clim(regridder):
    """Cached per-doy (threshold, mean, std) on the 2.8 deg grid, 1991-2020."""
    CLIM_DIR.mkdir(parents=True, exist_ok=True)
    path = CLIM_DIR / f"clim_w{WINDOW}_2p8deg.nc"
    if path.exists():
        print(f"[clim] cached {path}", flush=True)
        return xr.open_dataset(path)
    ref = sorted(f for f in glob.glob(f"{DATA}/t850_24h_world_*.nc")
                 if 1991 <= _year(f) <= 2020)
    print(f"[clim] building from {len(ref)} ref years (regrid 0.25->2.8 deg) ...",
          flush=True)
    vals, times, lat, lon = load_regridded(ref, regridder)
    doy = times.dayofyear.values
    Y, X = vals.shape[1:]
    ndoy = 366
    thr = np.full((ndoy, Y, X), np.nan, "float32")
    mean = np.full((ndoy, Y, X), np.nan, "float32")
    std = np.full((ndoy, Y, X), np.nan, "float32")
    for d in range(1, ndoy + 1):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, ndoy - dist)
        sel = vals[dist <= WINDOW]
        if sel.shape[0]:
            thr[d - 1] = np.quantile(sel, Q, axis=0)
            mean[d - 1] = sel.mean(0)
            std[d - 1] = sel.std(0)
        if d % 60 == 0:
            print(f"    doy {d}/{ndoy}", flush=True)
    ds = xr.Dataset(
        {"threshold": (("dayofyear", "latitude", "longitude"), thr),
         "mean":      (("dayofyear", "latitude", "longitude"), mean),
         "std":       (("dayofyear", "latitude", "longitude"), std)},
        coords={"dayofyear": np.arange(1, ndoy + 1), "latitude": lat,
                "longitude": lon})
    ds.attrs.update(window_halfwidth_days=WINDOW, quantile=Q,
                    reference="1991-2020", grid="2.8deg")
    enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    print(f"[clim] wrote {path} {dict(ds.sizes)}", flush=True)
    return ds


def active_and_z(vals, times, clim):
    """Per-cell hot-spell mask (>=MIN_DUR) + standardized anomaly z, for the region-
    cropped arrays. Returns active[T,Y,X] bool, z[T,Y,X]."""
    doy = times.dayofyear.values
    thr = clim["threshold"].values[doy - 1]
    mu = clim["mean"].values[doy - 1]
    sd = clim["std"].values[doy - 1]
    z = np.where(sd > 1e-6, (vals - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0)
    hot = vals > thr
    T, Y, X = hot.shape
    active = np.zeros((T, Y, X), bool)
    for j in range(Y):
        for i in range(X):
            col = hot[:, j, i]
            t = 0
            while t < T:
                if col[t]:
                    s = t
                    while t < T and col[t]:
                        t += 1
                    if t - s >= MIN_DUR:
                        active[s:t, j, i] = True
                else:
                    t += 1
    return active, z


_COAST = None


def draw_coast(ax):
    global _COAST
    try:
        if _COAST is None:
            cf = xr.open_zarr(_COAST_ZARR)
            m = cf["land_sea_mask"]
            m = m.assign_coords(lon=(((m.lon + 180) % 360) - 180)).sortby("lon").sortby("lat")
            _COAST = (m.lon.values, m.lat.values, m.values)
        lon, lat, mask = _COAST
        ax.contour(lon, lat, mask, levels=[0.5], colors="k", linewidths=0.5, alpha=0.7)
    except Exception:
        pass


def frame(lon2d, lat2d, z_day, active_day, date, w, e, s, n, region, nactive):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    norm = Normalize(-ZLIM, ZLIM)
    cmap = "RdBu_r"
    # faint full field (30% opacity)
    ax.pcolormesh(lon2d, lat2d, z_day, cmap=cmap, norm=norm, shading="auto", alpha=0.30)
    # affected cells at full opacity
    zc = np.where(active_day, z_day, np.nan)
    m = ax.pcolormesh(lon2d, lat2d, zc, cmap=cmap, norm=norm, shading="auto", alpha=1.0)
    draw_coast(ax)
    ax.set_xlim(w, e); ax.set_ylim(s, n)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"{region.title()} heat waves {date.date()}\n"
                 f"$\\pm${WINDOW}-day window, {nactive} cells in heat wave",
                 fontsize=11)
    cb = fig.colorbar(m, ax=ax, shrink=0.85)
    cb.set_label("T$_{850}$ anomaly  (σ from 1991-2020 daily mean)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    tgt = sorted(f for f in glob.glob(f"{DATA}/t850_24h_world_*.nc") if _year(f) == YEAR)
    if not tgt:
        raise SystemExit(f"{YEAR} T850 file missing in {DATA}")
    with xr.open_dataset(tgt[0]) as ds0:
        regridder = build_regridder(ds0["temperature"].isel(time=0))
    clim = build_or_load_clim(regridder)

    print(f"[anim] regridding target {YEAR} ...", flush=True)
    vals, times, lat, lon = load_regridded(tgt, regridder)

    # crop to region (both target + clim)
    latm, lonm = region_mask(lat, lon, REGION)
    vals = vals[:, latm][:, :, lonm]
    clim = clim.isel(latitude=latm, longitude=lonm)
    latc = lat[latm]
    lonc = ((lon[lonm] + 180) % 360) - 180
    order = np.argsort(lonc)                       # sort lon to -180..180 ascending
    lonc = lonc[order]; vals = vals[:, :, order]
    clim = clim.isel(longitude=order)
    lon2d, lat2d = np.meshgrid(lonc, latc)
    w, e, s, n = region_extent(REGION)

    active, z = active_and_z(vals, times, clim)
    day_has = active.any(axis=(1, 2))
    days = np.where(day_has)[0]
    print(f"[anim] {REGION}: {len(days)} days with an active heat wave in {YEAR}",
          flush=True)
    if len(days) == 0:
        raise SystemExit("no active heat-wave days -> no GIF")

    frames = []
    for k, t in enumerate(days):
        frames.append(frame(lon2d, lat2d, z[t], active[t], times[t],
                            w, e, s, n, REGION, int(active[t].sum())))
        if k % 10 == 0:
            print(f"    frame {k+1}/{len(days)} ({times[t].date()})", flush=True)

    out = GIF_DIR / f"heatwave{YEAR}_{REGION}_w{WINDOW}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=1000, loop=0, optimize=True)
    print(f"[anim] DONE -> {out} ({len(frames)} frames, 1 s each, "
          f"{out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
