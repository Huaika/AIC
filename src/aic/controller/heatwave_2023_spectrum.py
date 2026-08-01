#!/usr/bin/env python
"""2023 heat-wave DURATION SPECTRA across reference-window sizes (+/-0 .. +/-7 days),
on the NeuralGCM 2.8 deg model grid.

The 0.25 deg ERA5 T850 is CONSERVATIVELY REGRIDDED onto NeuralGCM's own 128x64
(~2.8 deg) grid -- the same grid + regridder the eval pipeline uses for truth -- so
the heat-wave results sit natively on the model grid and need NO interpolation when
compared to NeuralGCM later.

Gridded, global: at each model-grid cell the 1991-2020 day-of-year 95th-percentile
T850 threshold is built with a +/-w-day window; 2023 heat waves are the >=3-
consecutive-day spells above it. For every window w we plot the distribution of
heat-wave DURATION over all cells, two ways (one file each):

  * count : x = duration (consecutive days), y = NUMBER of heat waves (cell events).
  * area  : x = duration, y = total AREA affected (10^6 km^2) -- the "region size".

One PDF per (metric, window) -> 2 x 8 = 16 self-describing, publication-ready files,
with axes shared across the 8 windows of a metric for direct comparability.
"""
from __future__ import annotations
import glob
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

DATA = os.environ.get(
    "HW_DATA_DIR", "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave")
FIGDIR = Path(os.environ.get(
    "HW_FIG_DIR", "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_figures"))
Q = float(os.environ.get("HW_Q", "0.95"))
MIN_DUR = int(os.environ.get("HW_MIN_DURATION", "3"))
WINDOWS = list(range(0, 8))                        # +/-0 .. +/-7 days
YEAR = 2023
MODEL_NAME = "v1/deterministic_2_8_deg.pkl"        # NeuralGCM 2.8 deg checkpoint
REGRID_BATCH = int(os.environ.get("HW_REGRID_BATCH", "300"))
RES_TAG = "2p8deg"

COUNT_COLOR = "#2166ac"   # blue   (event count)
AREA_COLOR = "#b35806"    # orange (area / region size)
INK = "#222222"; MUTED = "#666666"; GRID = "#d9d9d9"


def _year(f):
    return int(os.path.basename(f).split("_")[-1].split(".")[0])


def build_regridder(sample):
    """Conservative regridder from the 0.25 deg source onto NeuralGCM's model grid
    (identical to eval_common._build_regridder)."""
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


def load_regridded(files, regridder):
    """Open the 0.25 deg T850, regrid to the model grid in time-batches (bounds
    memory), return (vals[T,Y,X] float32, doy[T], lat[Y])."""
    ds = xr.open_mfdataset(sorted(files), combine="by_coords")
    da = ds["temperature"]
    n = da.sizes["time"]
    chunks = []
    for s in range(0, n, REGRID_BATCH):
        sub = da.isel(time=slice(s, s + REGRID_BATCH)).compute()
        chunks.append(xarray_utils.regrid(sub, regridder))
        print(f"    regridded {min(s+REGRID_BATCH, n)}/{n}", flush=True)
    reg = xr.concat(chunks, dim="time")
    # the model grid comes back as (time, longitude, latitude); force a known order
    reg = reg.transpose("time", "latitude", "longitude")
    vals = reg.values.astype("float32")
    doy = pd.to_datetime(reg["time"].values).dayofyear.values.astype("int16")
    return vals, doy, reg["latitude"].values


def grid_cell_area_km2(lat, nlon):
    """Spherical cell area per latitude band on the model grid (km^2), from the
    latitude cell edges (handles the model grid's non-uniform lat spacing)."""
    lat = np.asarray(lat, float)
    edges = np.empty(len(lat) + 1)
    edges[1:-1] = (lat[:-1] + lat[1:]) / 2
    edges[0] = lat[0] - (lat[1] - lat[0]) / 2
    edges[-1] = lat[-1] + (lat[-1] - lat[-2]) / 2
    edges = np.clip(edges, -90, 90)
    R = 6371.0088
    dlon = 2 * np.pi / nlon
    band = (R ** 2) * dlon * np.abs(np.sin(np.deg2rad(edges[1:]))
                                    - np.sin(np.deg2rad(edges[:-1])))
    return band  # (Y,)


def doy_threshold_grid(vals, doy, w, q=Q, ndoy=366):
    Y, X = vals.shape[1:]
    thr = np.full((ndoy + 1, Y, X), np.nan, "float32")
    for d in range(1, ndoy + 1):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, ndoy - dist)
        sel = vals[dist <= w]
        if sel.shape[0]:
            thr[d] = np.quantile(sel, q, axis=0)
    return thr


def spell_events(tgt_vals, tgt_doy, thr, area_flat, min_dur=MIN_DUR):
    """Vectorised >=min_dur run detection along time for every cell -> (durations,
    per-event cell area)."""
    hot = (tgt_vals - thr[tgt_doy]) > 0
    T = hot.shape[0]
    hct = hot.reshape(T, -1).T.astype("int8")
    h = np.pad(hct, ((0, 0), (1, 1)))
    d = np.diff(h, axis=1)
    cs, ts = np.where(d == 1)
    ce, te = np.where(d == -1)
    length = te - ts
    keep = length >= min_dur
    return length[keep], area_flat[cs[keep]]


def publication_bar(x, y, color, title, ylabel, xlim, ylim, path, note):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x, y, width=0.82, color=color, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_title(title, fontsize=12, color=INK, loc="left", pad=10)
    ax.set_xlabel("Heat-wave duration (consecutive days)", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, fontsize=11, color=INK)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    step = max(1, (int(xlim[1]) // 15))
    ax.set_xticks(np.arange(int(xlim[0]) + 1, int(xlim[1]) + 1, step))
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.text(0.98, 0.95, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{DATA}/t850_24h_world_*.nc"))
    ref = [f for f in files if 1991 <= _year(f) <= 2020]
    tgt = [f for f in files if _year(f) == YEAR]
    if not tgt:
        raise SystemExit(f"{YEAR} T850 file missing in {DATA}")
    print(f"[spec] ref {len(ref)} yrs, target {YEAR}; regridding 0.25deg -> "
          f"NeuralGCM 2.8deg grid; windows +/-{WINDOWS}", flush=True)

    with xr.open_dataset(tgt[0]) as ds0:
        regridder = build_regridder(ds0["temperature"].isel(time=0))
    print("[spec] regridder built (NeuralGCM model grid)", flush=True)

    print("[spec] regridding reference (1991-2020) ...", flush=True)
    ref_vals, ref_doy, lat = load_regridded(ref, regridder)
    print("[spec] regridding target (2023) ...", flush=True)
    tgt_vals, tgt_doy, _ = load_regridded(tgt, regridder)
    Y, X = ref_vals.shape[1:]
    band = grid_cell_area_km2(lat, X)
    area_flat = np.repeat(band[:, None], X, axis=1).reshape(-1).astype("float64")
    print(f"[spec] model grid {Y}x{X} = {Y*X} cells; {ref_vals.shape[0]} ref days",
          flush=True)

    per_w = {}
    for w in WINDOWS:
        thr = doy_threshold_grid(ref_vals, ref_doy, w)
        dur, area = spell_events(tgt_vals, tgt_doy, thr, area_flat)
        per_w[w] = (dur, area)
        print(f"[spec] window +/-{w}d: {dur.size} heat-wave events, "
              f"{area.sum()/1e6:.2f} x10^6 km^2 total", flush=True)

    dmax = max((d.max() if d.size else MIN_DUR) for d, _ in per_w.values())
    xlim = (MIN_DUR - 0.6, dmax + 0.6)
    bins = np.arange(MIN_DUR, dmax + 1)
    cmax = 0.0; amax = 0.0; hist = {}
    for w, (dur, area) in per_w.items():
        c = np.array([(dur == b).sum() for b in bins])
        a = np.array([area[dur == b].sum() / 1e6 for b in bins])
        hist[w] = (c, a)
        cmax = max(cmax, c.max() if c.size else 0)
        amax = max(amax, a.max() if a.size else 0)
    cyl = (0, cmax * 1.08); ayl = (0, amax * 1.08)

    for w in WINDOWS:
        c, a = hist[w]
        base = (f"95th-pctile T$_{{850}}$ (00 UTC), 1991-2020 reference, "
                f"$\\pm${w}-day window, $\\geq${MIN_DUR}-day spells, "
                f"2.8$\\degree$ grid")
        publication_bar(
            bins, c, COUNT_COLOR,
            f"2023 heat-wave duration spectrum — event count\n{base}",
            "Number of heat waves (grid cells)", xlim, cyl,
            FIGDIR / f"heatwave{YEAR}_count-vs-duration_window-pm{w}d_{RES_TAG}.pdf",
            note=f"$\\pm${w} d window\n{int(c.sum())} heat waves")
        publication_bar(
            bins, a, AREA_COLOR,
            f"2023 heat-wave duration spectrum — area affected\n{base}",
            "Area affected (10$^6$ km$^2$)", xlim, ayl,
            FIGDIR / f"heatwave{YEAR}_area-vs-duration_window-pm{w}d_{RES_TAG}.pdf",
            note=f"$\\pm${w} d window\n{a.sum():.1f}$\\times$10$^6$ km$^2$")
        print(f"[spec] wrote window +/-{w}d (count + area)", flush=True)

    print(f"[spec] DONE -> {FIGDIR} (16 PDFs, {RES_TAG})", flush=True)


if __name__ == "__main__":
    main()
