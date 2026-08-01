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
  4. Keep only MAJOR-EVENT days: those where the major (99th) percentile is reached
     over >= HW_MAJOR_COVER of the region (widespread extreme, not a single cell) --
     so genuine major heat waves stand out. Split those days into temporally
     connected EPISODES -> one 1-second-per-frame GIF each (disconnected heat waves
     become separate gifs). Every frame: the WHOLE anomaly field faint (30% opacity)
     with the heat-wave (>=3-day spell) cells overdrawn at full opacity. All frames
     of all gifs share ONE colour palette, so the gradient/colouring is identical
     everywhere (no per-frame GIF-palette flicker).

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
try:
    from aic.view.gif_utils import shared_palette, quantize, save_gif
except ImportError:                                  # standalone run (work9)
    from gif_utils import shared_palette, quantize, save_gif

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
# only render days where at least this AREA fraction of the region is in a heat
# wave, so scattered single-cell days drop out and real episodes stand out.
COVER_FRAC = float(os.environ.get("HW_COVER_FRAC", "0.10"))
# temporally disconnected heat waves become SEPARATE gifs; qualifying days whose
# gap (in days) exceeds this start a new episode (2 -> tolerate a single-day dip).
EPISODE_GAP = int(os.environ.get("HW_EPISODE_GAP", "2"))
# only keep heat waves (>=3-day spells) that reach this climatological percentile
# somewhere -- "major" events. The whole spell containing such a day is kept, and a
# single cell is enough (no area-coverage requirement). Set HW_MAJOR_Q to disable
# with 0 (then any >=3-day spell counts).
MAJOR_Q = float(os.environ.get("HW_MAJOR_Q", "0.99"))
# a day is a MAJOR-EVENT day only if the major percentile is reached over at least
# this AREA fraction of the region (widespread extreme, not a single cell); those
# days define the episodes -> genuine major heat waves stand out.
MAJOR_COVER = float(os.environ.get("HW_MAJOR_COVER", "0.05"))
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


_REGRIDDER = None


def _get_regridder(sample_file):
    global _REGRIDDER
    if _REGRIDDER is None:
        with xr.open_dataset(sample_file) as ds0:
            _REGRIDDER = build_regridder(ds0["temperature"].isel(time=0))
    return _REGRIDDER


def regridded_field(name, files):
    """Cached regridded T850 (2.8 deg) for a set of yearly files. Regrids once
    (building the regridder on demand) and caches to CLIM_DIR/regrid_<name>_2p8deg.nc,
    so changing the window (or the year) never re-pays the 0.25->2.8 deg regrid."""
    CLIM_DIR.mkdir(parents=True, exist_ok=True)
    path = CLIM_DIR / f"regrid_{name}_2p8deg.nc"
    if path.exists():
        print(f"[regrid] cached {path.name}", flush=True)
        da = xr.open_dataarray(path)
        out = (da.values.astype("float32"), pd.to_datetime(da["time"].values),
               da["latitude"].values, da["longitude"].values)
        da.close()
        return out
    vals, times, lat, lon = load_regridded(files, _get_regridder(files[0]))
    xr.DataArray(vals, dims=("time", "latitude", "longitude"),
                 coords={"time": times.values, "latitude": lat, "longitude": lon},
                 name="temperature").to_netcdf(
        path, encoding={"temperature": {"dtype": "float32", "zlib": True,
                                        "complevel": 4}})
    print(f"[regrid] wrote {path.name}", flush=True)
    return vals, times, lat, lon


def build_or_load_clim():
    """Cached per-doy (threshold, threshold_major, mean, std) on the 2.8 deg grid,
    1991-2020. Uses the cached regridded reference, so a new window only recomputes
    the day-of-year percentiles (no re-regrid)."""
    CLIM_DIR.mkdir(parents=True, exist_ok=True)
    path = CLIM_DIR / f"clim_w{WINDOW}_2p8deg.nc"
    if path.exists():
        ds = xr.open_dataset(path)
        if ("threshold_major" in ds
                and abs(float(ds.attrs.get("major_quantile", -1)) - MAJOR_Q) < 1e-9):
            print(f"[clim] cached {path}", flush=True)
            return ds
        ds.close()
        print("[clim] cache missing/mismatched threshold_major -> rebuild", flush=True)
    ref = sorted(f for f in glob.glob(f"{DATA}/t850_24h_world_*.nc")
                 if 1991 <= _year(f) <= 2020)
    print(f"[clim] building day-of-year percentiles from {len(ref)} ref years ...",
          flush=True)
    vals, times, lat, lon = regridded_field("ref", ref)
    doy = times.dayofyear.values
    Y, X = vals.shape[1:]
    ndoy = 366
    thr = np.full((ndoy, Y, X), np.nan, "float32")
    thr_major = np.full((ndoy, Y, X), np.nan, "float32")
    mean = np.full((ndoy, Y, X), np.nan, "float32")
    std = np.full((ndoy, Y, X), np.nan, "float32")
    for d in range(1, ndoy + 1):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, ndoy - dist)
        sel = vals[dist <= WINDOW]
        if sel.shape[0]:
            thr[d - 1] = np.quantile(sel, Q, axis=0)
            thr_major[d - 1] = np.quantile(sel, MAJOR_Q, axis=0)
            mean[d - 1] = sel.mean(0)
            std[d - 1] = sel.std(0)
        if d % 60 == 0:
            print(f"    doy {d}/{ndoy}", flush=True)
    ds = xr.Dataset(
        {"threshold":       (("dayofyear", "latitude", "longitude"), thr),
         "threshold_major": (("dayofyear", "latitude", "longitude"), thr_major),
         "mean":            (("dayofyear", "latitude", "longitude"), mean),
         "std":             (("dayofyear", "latitude", "longitude"), std)},
        coords={"dayofyear": np.arange(1, ndoy + 1), "latitude": lat,
                "longitude": lon})
    ds.attrs.update(window_halfwidth_days=WINDOW, quantile=Q, major_quantile=MAJOR_Q,
                    reference="1991-2020", grid="2.8deg")
    enc = {v: {"dtype": "float32", "zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    print(f"[clim] wrote {path} {dict(ds.sizes)}", flush=True)
    return ds


def detect(vals, times, clim):
    """Per-cell >=MIN_DUR hot spells above the detection (95th) threshold -> active;
    cells at/above the major (99th) climatological threshold -> ext; standardized
    anomaly -> z. Returns active[T,Y,X], ext[T,Y,X], z[T,Y,X]."""
    doy = times.dayofyear.values
    thr = clim["threshold"].values[doy - 1]
    thr_m = clim["threshold_major"].values[doy - 1]
    mu = clim["mean"].values[doy - 1]
    sd = clim["std"].values[doy - 1]
    z = np.where(sd > 1e-6, (vals - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0)
    hot = vals > thr
    ext = vals > thr_m
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
    return active, ext, z


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


def frame(lon2d, lat2d, z_day, active_day, date, w, e, s, n, region, nactive,
          cover_frac):
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
                 f"$\\pm${WINDOW}-day window · {nactive} cells in heat wave · "
                 f"{cover_frac:.0%} at 99th pctile", fontsize=10.3)
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
    clim = build_or_load_clim()

    print(f"[anim] target {YEAR} ...", flush=True)
    vals, times, lat, lon = regridded_field(str(YEAR), tgt)

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

    active, ext, z = detect(vals, times, clim)
    aw = np.cos(np.deg2rad(latc))
    aw2d = np.repeat(aw[:, None], vals.shape[2], axis=1)          # (Yc, Xc)
    ext_cover = (ext * aw2d[None, :, :]).sum(axis=(1, 2)) / aw2d.sum()
    # a MAJOR-EVENT day: the major (99th) percentile is reached over >= MAJOR_COVER
    # of the region (widespread extreme). Episodes are runs of such days; frames
    # highlight the heat-wave (>=3-day spell) cells on those days.
    qual = np.where(ext_cover >= MAJOR_COVER)[0]
    print(f"[anim] {REGION}: {len(qual)} major-event days (>= {MAJOR_COVER:.0%} of "
          f"region at >= {MAJOR_Q:.0%}-pctile) in {YEAR} (vs "
          f"{int(active.any(axis=(1, 2)).sum())} days with any heat wave)", flush=True)
    if len(qual) == 0:
        raise SystemExit("no major-event days -> no GIF")

    # group qualifying days into temporally connected EPISODES (each -> its own gif);
    # an episode spans its first..last qualifying day so its animation is continuous.
    episodes = [[int(qual[0])]]
    for d in qual[1:]:
        (episodes[-1].append(int(d)) if d - episodes[-1][-1] <= EPISODE_GAP
         else episodes.append([int(d)]))
    spans = [list(range(ep[0], ep[-1] + 1)) for ep in episodes]
    print(f"[anim] {len(spans)} major heat-wave episode(s): " + ", ".join(
        f"{times[s[0]].date()}..{times[s[-1]].date()} ({len(s)}d)" for s in spans),
        flush=True)

    # render each needed frame ONCE (RGB), keyed by day index (highlight MAJOR cells)
    need = sorted({t for span in spans for t in span})
    rgb = {}
    for k, t in enumerate(need):
        rgb[t] = frame(lon2d, lat2d, z[t], active[t], times[t],
                       w, e, s, n, REGION, int(active[t].sum()), ext_cover[t])
        if k % 10 == 0:
            print(f"    rendered {k+1}/{len(need)}", flush=True)

    # ONE palette shared by every frame of every gif -> identical gradient/colouring
    pal = shared_palette([rgb[t] for t in need])
    quant = {t: quantize(rgb[t], pal) for t in need}

    mtag = f"_major{int(MAJOR_Q * 100)}c{int(MAJOR_COVER * 100)}"
    for span in spans:
        a, b = times[span[0]].date(), times[span[-1]].date()
        out = GIF_DIR / f"heatwave{YEAR}_{REGION}_w{WINDOW}{mtag}_{a}_{b}.gif"
        save_gif([quant[t] for t in span], out)
        print(f"[anim] wrote {out.name} ({len(span)} frames, "
              f"{out.stat().st_size / 1e6:.2f} MB)", flush=True)
    print(f"[anim] DONE -> {len(spans)} GIF(s) in {GIF_DIR}", flush=True)


if __name__ == "__main__":
    main()
