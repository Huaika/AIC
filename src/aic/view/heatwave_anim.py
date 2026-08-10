#!/usr/bin/env python
"""Animate detected heatwaves on a regional map (GIF), coloured by the day's
standardized T850 anomaly, for the best-performing reference window (+/-5 days).

Pipeline:
  1. Build (or load) a cached day-of-year CLIMATOLOGY on the 2.8 deg grid from
     1991-2020: per calendar day (pseudo-year, 1..366) and grid cell, the 95th-
     percentile heatwave THRESHOLD, the MEAN and the STD over a +/-window window.
     Saved once to clim_w<window>_2p8deg.nc -> future animations skip the 30-yr
     regrid entirely (this is the expensive step).
  2. Regrid the target year (2023) to the 2.8 deg grid, crop to the region.
  3. Per cell: hot = T850 > threshold[doy]; z = (T850 - mean[doy]) / std[doy];
     a cell is "affected" on days inside a >= 3-consecutive-hot-day spell.
  4. Keep only MAJOR-EVENT days: those where the major (99th) percentile is reached
     over >= HW_MAJOR_COVER of the region (widespread extreme, not a single cell) --
     so genuine major heatwaves stand out. Split those days into temporally
     connected EPISODES -> one 1-second-per-frame GIF each (disconnected heatwaves
     become separate gifs). Every frame: the WHOLE anomaly field faint (30% opacity)
     with the heatwave (>=3-day spell) cells overdrawn at full opacity. All frames
     of all gifs share ONE colour palette, so the gradient/colouring is identical
     everywhere (no per-frame GIF-palette flicker).

Env: HW_WINDOW (default 5), HW_SPEC_REGION (default europe), HW_YEAR (default 2023).
Only the region's cells drive the frame list. Coloured by (T-mean)/std (RdBu_r).
"""
from __future__ import annotations
import glob
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image

from aic import config
from aic.regions import REGIONS, region_mask, region_extent, wrap180
from aic.view.plotting import draw_coastlines
# shared analysis core (no duplication of regrid / thresholds / spell detection)
from aic.controller.heatwave.grid import regrid_da
from aic.controller.heatwave.climatology import doy_percentile
from aic.controller.heatwave.detect import active_mask
try:
    from aic.view.gif_utils import shared_palette, quantize, save_gif
except ImportError:                                  # standalone run (work9)
    from gif_utils import shared_palette, quantize, save_gif

DATA = config.env_str("HW_DATA_DIR", config.ERA5_HEATWAVE)
CLIM_DIR = Path(config.env_str("HW_CLIM_DIR", config.HEATWAVE_CLIM))
GIF_DIR = Path(config.env_str("HW_GIF_DIR", config.HEATWAVE_GIFS))
WINDOW = config.env_int("HW_WINDOW", 5)              # best-performing window
REGION = config.env_str("HW_SPEC_REGION", "europe").strip().lower()
YEAR = config.env_int("HW_YEAR", 2023)
# HW_CS_DEF: drive the affected-region SET (which cells, which days) from a named
# heatwave definition (e.g. "cordex") instead of this module's own T850 detection.
# The animation still colours by the T850 anomaly -- but relative to THAT definition's
# reference period (cordex: 1971-2000; others: 1991-2020), read from the 2.8deg
# daily-stats caches (t850 00-UTC value).
CS_DEF = config.env_str("HW_CS_DEF", "").strip()
DAILY_DIR = config.env_str("HW_DAILY_DIR", config.ERA5_HEATWAVE_DAILY)
REF_LABEL = "1991-2020"          # anomaly reference period (set per definition in main)
MIN_DUR = config.env_int("HW_MIN_DURATION", 3)
Q = config.env_float("HW_Q", 0.95)
REGRID_BATCH = config.env_int("HW_REGRID_BATCH", 300)
ZLIM = config.env_float("HW_ZLIM", 4.0)              # colour scale +/- sigma
# only render days where at least this AREA fraction of the region is in a heat
# wave, so scattered single-cell days drop out and real episodes stand out.
COVER_FRAC = config.env_float("HW_COVER_FRAC", 0.10)
# temporally disconnected heatwaves become SEPARATE gifs; qualifying days whose
# gap (in days) exceeds this start a new episode (2 -> tolerate a single-day dip).
EPISODE_GAP = config.env_int("HW_EPISODE_GAP", 2)
# only keep heatwaves (>=3-day spells) that reach this climatological percentile
# somewhere -- "major" events. The whole spell containing such a day is kept, and a
# single cell is enough (no area-coverage requirement). Set HW_MAJOR_Q to disable
# with 0 (then any >=3-day spell counts).
MAJOR_Q = config.env_float("HW_MAJOR_Q", 0.99)
# a day is a MAJOR-EVENT day only if the major percentile is reached over at least
# this AREA fraction of the region (widespread extreme, not a single cell); those
# days define the episodes -> genuine major heatwaves stand out.
MAJOR_COVER = config.env_float("HW_MAJOR_COVER", 0.05)


def _year(f):
    return int(os.path.basename(f).split("_")[-1].split(".")[0])


def regridded_field(name, files):
    """Cached regridded T850 (2.8 deg) for a set of yearly files -- regrid once (via
    the shared grid.regrid_da) and cache to CLIM_DIR/regrid_<name>_2p8deg.nc, so a new
    window/year never re-pays the 0.25->2.8 deg regrid."""
    CLIM_DIR.mkdir(parents=True, exist_ok=True)
    path = CLIM_DIR / f"regrid_{name}_2p8deg.nc"
    if path.exists():
        print(f"[regrid] cached {path.name}", flush=True)
        da = xr.open_dataarray(path)
        out = (da.values.astype("float32"), pd.to_datetime(da["time"].values),
               da["latitude"].values, da["longitude"].values)
        da.close()
        return out
    ds = xr.open_mfdataset(sorted(files), combine="by_coords")
    reg = regrid_da(ds["temperature"], batch=REGRID_BATCH)
    vals = reg.values.astype("float32")
    times = pd.to_datetime(reg["time"].values)
    lat = reg["latitude"].values; lon = reg["longitude"].values
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
    thr = doy_percentile(vals, doy, WINDOW, Q)              # shared core
    thr_major = doy_percentile(vals, doy, WINDOW, MAJOR_Q)
    mean = np.full((ndoy, Y, X), np.nan, "float32")
    std = np.full((ndoy, Y, X), np.nan, "float32")
    for d in range(1, ndoy + 1):                            # mean/std for the z-score
        dist = np.abs(doy - d)
        dist = np.minimum(dist, ndoy - dist)
        sel = vals[dist <= WINDOW]
        if sel.shape[0]:
            mean[d - 1] = sel.mean(0)
            std[d - 1] = sel.std(0)
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
    active = active_mask(hot, MIN_DUR)                      # shared core
    return active, ext, z


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
    draw_coastlines(ax, lw=0.5)
    ax.set_xlim(w, e); ax.set_ylim(s, n)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    sub = (f"{CS_DEF.upper()} heatwave footprint · {nactive} cells · "
           f"{cover_frac:.0%} of region" if CS_DEF else
           f"$\\pm${WINDOW}-day window · {nactive} cells in heatwave · "
           f"{cover_frac:.0%} at 99th pctile")
    ax.set_title(f"{region.title()} heatwaves {date.date()}\n{sub}", fontsize=10.3)
    cb = fig.colorbar(m, ax=ax, shrink=0.85)
    cb.set_label(f"T$_{{850}}$ anomaly  (σ from {REF_LABEL} daily mean)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _doy_mean_std(vals, doy, window, ndoy=366):
    """Per-calendar-day mean and std of `vals` (T,Y,X), pooling a circular +/-window."""
    Y, X = vals.shape[1:]
    mean = np.full((ndoy, Y, X), np.nan, "float32")
    std = np.full((ndoy, Y, X), np.nan, "float32")
    for d in range(1, ndoy + 1):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, ndoy - dist)
        sel = vals[dist <= window]
        if sel.shape[0]:
            mean[d - 1] = sel.mean(0)
            std[d - 1] = sel.std(0)
    return mean, std


def def_reference(defn, window):
    """The colouring reference for a definition: (mean, std) day-of-year climatology
    of 00-UTC T850 over the definition's OWN reference period (e.g. 1971-2000 for
    cordex), plus the target-year field -- all from the 2.8deg daily-stats caches
    (t850 'tval'), so no re-regrid. Returns (mean, std, vals, times, lat, lon)."""
    from aic.controller.heatwave.grid import load_daily_regridded
    r0, r1 = defn.ref_years
    cache = CLIM_DIR / f"clim_{defn.name}_{r0}-{r1}_w{window}_2p8deg.nc"
    if cache.exists():
        cd = xr.open_dataset(cache)
        mean, std = cd["mean"].values, cd["std"].values
        lat, lon = cd["latitude"].values, cd["longitude"].values
        cd.close()
    else:
        refs = [load_daily_regridded(DAILY_DIR, "t850", y, ["tval"], str(CLIM_DIR))
                for y in defn.ref_range]
        ref = xr.concat(refs, dim="time")
        vals = ref["t850_tval"].values.astype("float32")
        doy = pd.to_datetime(ref["time"].values).dayofyear.values
        lat = ref["latitude"].values; lon = ref["longitude"].values
        mean, std = _doy_mean_std(vals, doy, window)
        xr.Dataset({"mean": (("dayofyear", "latitude", "longitude"), mean),
                    "std":  (("dayofyear", "latitude", "longitude"), std)},
                   coords={"dayofyear": np.arange(1, 367), "latitude": lat,
                           "longitude": lon}).to_netcdf(cache)
        print(f"[clim] wrote {cache.name} ({r0}-{r1} T850 ref)", flush=True)
    tds = load_daily_regridded(DAILY_DIR, "t850", YEAR, ["tval"], str(CLIM_DIR))
    return (mean, std, tds["t850_tval"].values.astype("float32"),
            pd.to_datetime(tds["time"].values), lat, lon)


def main():
    global REF_LABEL
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    if CS_DEF:
        # SET of affected (cell, day) + the anomaly reference come from a named
        # definition (e.g. cordex): the colouring is the T850 anomaly relative to
        # THAT definition's reference period (cordex 1971-2000), and the highlighted
        # cells/episodes come from its active mask.
        from aic.controller.casestudy import heatwave_mask as HM
        from aic.controller.heatwave import definitions as D
        defn = D.BY_NAME[CS_DEF]
        REF_LABEL = f"{defn.ref_years[0]}-{defn.ref_years[1]}"
        print(f"[anim] {CS_DEF} target {YEAR} (T850 anomaly vs {REF_LABEL}) ...", flush=True)
        mean, std, vals, times, lat, lon = def_reference(defn, WINDOW)
    else:
        tgt = sorted(f for f in glob.glob(f"{DATA}/t850_24h_world_*.nc") if _year(f) == YEAR)
        if not tgt:
            raise SystemExit(f"{YEAR} T850 file missing in {DATA}")
        clim = build_or_load_clim()
        print(f"[anim] target {YEAR} ...", flush=True)
        vals, times, lat, lon = regridded_field(str(YEAR), tgt)

    # crop to region + sort lon to -180..180 ascending
    latm, lonm = region_mask(lat, lon, REGION)
    latc = lat[latm]
    lonc = wrap180(lon[lonm])
    order = np.argsort(lonc)
    lonc = lonc[order]
    crop = lambda a: a[:, latm][:, :, lonm][:, :, order]
    vals = crop(vals)
    lon2d, lat2d = np.meshgrid(lonc, latc)
    w, e, s, n = region_extent(REGION)

    if CS_DEF:
        doy = times.dayofyear.values
        mu = crop(mean)[doy - 1]; sd = crop(std)[doy - 1]
        z = np.where(sd > 1e-6, (vals - mu) / np.where(sd > 1e-6, sd, 1.0), 0.0)
        ad = HM.active_mask_da(defn, YEAR, region=REGION).reindex(
            time=times.values, method="nearest")           # (T, 64, 128), same grid
        active = ad.values[:, latm][:, :, lonm][:, :, order]
        ext_cover = HM.region_coverage(ad, REGION)          # per-day area fraction
        eps = HM.episodes(ad, REGION)
        if not eps:
            raise SystemExit(f"no {CS_DEF} episodes in {REGION} {YEAR} -> no GIF")
        spans = [list(range(e.span[0], e.span[1] + 1)) for e in eps]
        # each event highlights ONLY its own (spatio-temporally connected) cells
        ep_hl = [e.mask[:, latm][:, :, lonm][:, :, order] for e in eps]
        print(f"[anim] {CS_DEF} ({defn.label}): {len(spans)} spatio-temporal event(s): "
              + ", ".join(f"{times[s[0]].date()}..{times[s[-1]].date()}"
                          f"({len(s)}d,{int(e.n_cells)}c)" for s, e in zip(spans, eps)),
              flush=True)
    else:
        clim = clim.isel(latitude=latm, longitude=lonm).isel(longitude=order)
        active, ext, z = detect(vals, times, clim)      # z = T850 anomaly (colouring)
        aw = np.cos(np.deg2rad(latc))
        aw2d = np.repeat(aw[:, None], vals.shape[2], axis=1)          # (Yc, Xc)
        ext_cover = (ext * aw2d[None, :, :]).sum(axis=(1, 2)) / aw2d.sum()
        # a MAJOR-EVENT day: the major (99th) percentile is reached over >= MAJOR_COVER
        # of the region (widespread extreme). Episodes are runs of such days.
        qual = np.where(ext_cover >= MAJOR_COVER)[0]
        print(f"[anim] {REGION}: {len(qual)} major-event days (>= {MAJOR_COVER:.0%} of "
              f"region at >= {MAJOR_Q:.0%}-pctile) in {YEAR} (vs "
              f"{int(active.any(axis=(1, 2)).sum())} days with any heatwave)", flush=True)
        if len(qual) == 0:
            raise SystemExit("no major-event days -> no GIF")
        episodes = [[int(qual[0])]]
        for d in qual[1:]:
            (episodes[-1].append(int(d)) if d - episodes[-1][-1] <= EPISODE_GAP
             else episodes.append([int(d)]))
        spans = [list(range(ep[0], ep[-1] + 1)) for ep in episodes]
        ep_hl = [active for _ in spans]     # time-based episodes share the daily mask
        print(f"[anim] {len(spans)} major heatwave episode(s): " + ", ".join(
            f"{times[s[0]].date()}..{times[s[-1]].date()} ({len(s)}d)" for s in spans),
            flush=True)

    # render each (event, day) frame ONCE, highlighting THAT event's cells. The
    # subtitle coverage is the event's OWN area fraction (matches its cell count) for
    # a definition-driven run, else the day's 99th-pctile coverage (the "ours" mode).
    awc2d = np.repeat(np.cos(np.deg2rad(latc))[:, None], len(lonc), axis=1)
    reg_area = awc2d.sum()
    rgb = {}
    for ei, span in enumerate(spans):
        hlm = ep_hl[ei]
        for t in span:
            hl = hlm[t]
            cov = float((hl * awc2d).sum() / reg_area) if CS_DEF else ext_cover[t]
            rgb[(ei, t)] = frame(lon2d, lat2d, z[t], hl, times[t], w, e, s, n,
                                 REGION, int(hl.sum()), cov)
    print(f"    rendered {len(rgb)} frame(s) over {len(spans)} event(s)", flush=True)

    # ONE palette shared by every frame of every gif -> identical gradient/colouring
    keys = list(rgb)
    pal = shared_palette([rgb[k] for k in keys])
    quant = {k: quantize(rgb[k], pal) for k in keys}

    # one folder per definition (cordex / ours / ...)
    subdir = GIF_DIR / (CS_DEF or "ours")
    subdir.mkdir(parents=True, exist_ok=True)
    tag = "" if CS_DEF else f"_w{WINDOW}_major{int(MAJOR_Q*100)}c{int(MAJOR_COVER*100)}"
    for ei, span in enumerate(spans):
        a, b = times[span[0]].date(), times[span[-1]].date()
        # ei in the name: distinct spatial events can share a day range
        out = subdir / f"heatwave{YEAR}_{REGION}{tag}_ev{ei+1:02d}_{a}_{b}.gif"
        save_gif([quant[(ei, t)] for t in span], out)
        print(f"[anim] wrote {out.name} ({len(span)} frames, "
              f"{out.stat().st_size / 1e6:.2f} MB)", flush=True)
    print(f"[anim] DONE -> {len(spans)} GIF(s) in {subdir}", flush=True)


if __name__ == "__main__":
    main()
