#!/usr/bin/env python
"""Heat-wave DURATION SPECTRA for our T850-00 UTC definition, across window sizes
(+/-0..+/-7), on the NeuralGCM 2.8 deg grid.

The regrid, day-of-year percentile threshold, cell area and spell detection all come
from the shared controller/heatwave core (grid / climatology / detect) -- this module
only reads the T850 00 UTC files, drives the window loop and draws the bar charts.

  * count : x = duration (days), y = number of heat waves (grid-cell events).
  * area  : x = duration, y = total area affected (10^6 km^2).
2 x 8 = 16 PDFs, axes shared across windows for direct comparability.
"""
from __future__ import annotations
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aic.regions import REGIONS, region_mask
from aic.controller.heatwave.grid import build_regridder, regrid_da
from aic.controller.heatwave.climatology import doy_percentile
from aic.controller.heatwave.detect import spell_events, cell_area_km2, MIN_DUR

DATA = os.environ.get(
    "HW_DATA_DIR", "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave")
FIGDIR = Path(os.environ.get(
    "HW_FIG_DIR", "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_figures"))
Q = float(os.environ.get("HW_Q", "0.95"))
WINDOWS = list(range(0, 8))
YEAR = 2023
REGRID_BATCH = int(os.environ.get("HW_REGRID_BATCH", "300"))
RES_TAG = "2p8deg"
REGION = os.environ.get("HW_SPEC_REGION", "world").strip().lower()

COUNT_COLOR = "#2166ac"; AREA_COLOR = "#b35806"
from aic.view.plotting import INK, GRID, MUTED  # shared palette (single source)


def _year(f):
    return int(os.path.basename(f).split("_")[-1].split(".")[0])


def _regrid(files):
    """Open the 0.25 deg T850 files and regrid to the model grid (batched)."""
    ds = xr.open_mfdataset(sorted(files), combine="by_coords")
    reg = regrid_da(ds["temperature"], batch=REGRID_BATCH)
    return (reg.values.astype("float32"),
            pd.to_datetime(reg["time"].values).dayofyear.values.astype("int16"),
            reg["latitude"].values, reg["longitude"].values)


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
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{DATA}/t850_24h_world_*.nc"))
    ref = [f for f in files if 1991 <= _year(f) <= 2020]
    tgt = [f for f in files if _year(f) == YEAR]
    if not tgt:
        raise SystemExit(f"{YEAR} T850 file missing in {DATA}")
    if REGION not in REGIONS:
        raise SystemExit(f"HW_SPEC_REGION must be one of {list(REGIONS)}")
    print(f"[spec] regridding ref ({len(ref)} yrs) + target {YEAR}; windows +/-{WINDOWS}",
          flush=True)
    ref_vals, ref_doy, lat, lon = _regrid(ref)
    tgt_vals, tgt_doy, _, _ = _regrid(tgt)
    nlon_full = ref_vals.shape[2]

    if REGION != "world":
        latm, lonm = region_mask(lat, lon, REGION)
        ref_vals = ref_vals[:, latm][:, :, lonm]
        tgt_vals = tgt_vals[:, latm][:, :, lonm]
        lat = lat[latm]
    Y, X = ref_vals.shape[1:]
    band = cell_area_km2(lat, nlon_full)               # full-grid dlon
    area_flat = np.repeat(band[:, None], X, axis=1).reshape(-1).astype("float64")
    print(f"[spec] grid {Y}x{X} = {Y*X} cells ({REGION}); {ref_vals.shape[0]} ref days",
          flush=True)

    per_w = {}
    for w in WINDOWS:
        thr = doy_percentile(ref_vals, ref_doy, w, Q)  # (ndoy, Y, X)
        hot = tgt_vals > thr[tgt_doy - 1]
        dur, area = spell_events(hot, area_flat)
        per_w[w] = (dur, area)
        print(f"[spec] window +/-{w}d: {dur.size} heat-wave events, "
              f"{area.sum()/1e6:.2f} x10^6 km^2 total", flush=True)

    dmax = max((d.max() if d.size else MIN_DUR) for d, _ in per_w.values())
    xlim = (MIN_DUR - 0.6, dmax + 0.6)
    bins = np.arange(MIN_DUR, dmax + 1)
    cmax = amax = 0.0; hist = {}
    for w, (dur, area) in per_w.items():
        c = np.array([(dur == b).sum() for b in bins])
        a = np.array([area[dur == b].sum() / 1e6 for b in bins])
        hist[w] = (c, a)
        cmax = max(cmax, c.max() if c.size else 0)
        amax = max(amax, a.max() if a.size else 0)
    cyl = (0, cmax * 1.08); ayl = (0, amax * 1.08)
    rsuf = "" if REGION == "world" else f"_{REGION}"
    rlabel = "" if REGION == "world" else f"{REGION.replace('_', ' ').title()} — "

    for w in WINDOWS:
        c, a = hist[w]
        base = (f"95th-pctile T$_{{850}}$ (00 UTC), 1991-2020 reference, "
                f"$\\pm${w}-day window, $\\geq${MIN_DUR}-day spells, 2.8$\\degree$ grid")
        publication_bar(
            bins, c, COUNT_COLOR,
            f"{rlabel}2023 heat-wave duration spectrum — event count\n{base}",
            "Number of heat waves (grid cells)", xlim, cyl,
            FIGDIR / f"heatwave{YEAR}_count-vs-duration_window-pm{w}d_{RES_TAG}{rsuf}.pdf",
            note=f"$\\pm${w} d window\n{int(c.sum())} heat waves")
        publication_bar(
            bins, a, AREA_COLOR,
            f"{rlabel}2023 heat-wave duration spectrum — area affected\n{base}",
            "Area affected (10$^6$ km$^2$)", xlim, ayl,
            FIGDIR / f"heatwave{YEAR}_area-vs-duration_window-pm{w}d_{RES_TAG}{rsuf}.pdf",
            note=f"$\\pm${w} d window\n{a.sum():.2f}$\\times$10$^6$ km$^2$")
    print(f"[spec] DONE -> {FIGDIR} (16 PDFs, {RES_TAG}{rsuf})", flush=True)


if __name__ == "__main__":
    main()
