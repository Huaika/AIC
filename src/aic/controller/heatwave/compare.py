#!/usr/bin/env python
"""Compare the three heat-wave DEFINITIONS (ours / mixture / ECMWF) on the model
grid, ACROSS a sweep of +/-window sizes, for a target year.

Colour encodes the definition (blue=ours, purple=mixture, red=ECMWF); the SHADE
encodes the window size -- darker = smaller window, lighter = larger.

Two families of figures (each overlays all definitions x windows):
  1. amount vs duration : x = heat-wave duration (days), y = count / area affected.
  2. timing over the year: x = day of year (labelled by month, leap-safe), y = area
     of the region in a heat wave on that day -- when the heat waves happen.

Env: HW_WINDOWS (default "0,1,3,5,7"), HW_SPEC_REGION (default world),
HW_YEAR (default 2023), HW_DEFS, HW_DAILY_DIR, HW_CACHE_DIR, HW_FIG_DIR.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from aic import config
from aic.controller.heatwave import definitions as D
from aic.controller.heatwave import detect as DET
from aic.controller.heatwave.grid import load_daily_regridded
from aic.regions import region_mask

DAILY_DIR = config.env_str("HW_DAILY_DIR", config.ERA5_HEATWAVE_DAILY)
CACHE_DIR = config.env_str("HW_CACHE_DIR", config.HEATWAVE_CLIM)
FIGDIR = Path(config.env_str("HW_FIG_DIR", config.HEATWAVE_FIGURES))
WINDOWS = [int(x) for x in config.env_list("HW_WINDOWS", ["0", "1", "3", "5", "7"])]
REGION = config.env_str("HW_SPEC_REGION", "world").strip().lower()
YEAR = config.env_int("HW_YEAR", 2023)
REGION_NAME = "the world" if REGION == "world" else REGION.replace("_", " ").title()
REF = range(1991, 2021)
# descriptive definition labels (variable + cadence), shared by all comparison plots
DEF_LABELS = {"ours": "850hPa 00 UTC", "mixture": "850hPa 6-hourly",
              "ecmwf": "2mT 6-hourly",
              "cordex": "2mT max, May–Sep p99 (1971–2000)"}
from aic.style import INK, GRID  # shared palette (single source of truth)


def shades(base_hex, n, fmax=0.66):
    """n colours from the base hue (dark, f=0) to a light tint (f=fmax, toward white)."""
    base = np.array(mcolors.to_rgb(base_hex))
    return [tuple(base * (1 - f) + f) for f in np.linspace(0.0, fmax, n)]


def _load_key(tag, stats, ref_years):
    """Ref (concat ref_years, inclusive) + target regridded daily stats for a
    (variable, reference-period), cropped to REGION ->
    (ref_stats, ref_doy, ref_months, tgt_stats, tgt_doy, tgt_months, latc, nlon)."""
    ref = xr.concat([load_daily_regridded(DAILY_DIR, tag, y, stats, CACHE_DIR)
                     for y in range(ref_years[0], ref_years[1] + 1)], dim="time")
    tgt = load_daily_regridded(DAILY_DIR, tag, YEAR, stats, CACHE_DIR)
    lat = ref["latitude"].values; lon = ref["longitude"].values
    latm, lonm = (region_mask(lat, lon, REGION) if REGION != "world"
                  else (np.ones(len(lat), bool), np.ones(len(lon), bool)))
    arrs = lambda ds: {s: ds[f"{tag}_{s}"].values[:, latm][:, :, lonm] for s in stats}
    rt = pd.to_datetime(ref["time"].values); tt = pd.to_datetime(tgt["time"].values)
    return (arrs(ref), rt.dayofyear.values, rt.month.values,
            arrs(tgt), tt.dayofyear.values, tt.month.values,
            lat[latm], int(lonm.sum()))


def month_ticks(year):
    ms = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    return ms.dayofyear.values, [m.strftime("%b") for m in ms]


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    defs = D.selected(config.env_str("HW_DEFS"))
    # load ref+target once per (tag, reference-period); union the stats each key needs
    keys = {}
    for d in defs:
        keys.setdefault((d.tag, d.ref_years), set()).update(d.stats)
    print(f"[compare] {YEAR} region={REGION} windows={WINDOWS} defs={[d.name for d in defs]}",
          flush=True)
    data = {k: _load_key(k[0], sorted(sts), k[1]) for k, sts in keys.items()}

    d0 = data[(defs[0].tag, defs[0].ref_years)]
    latc, nlon = d0[6], d0[7]
    band = DET.cell_area_km2(latc, 128)
    area2d = np.repeat(band[:, None], nlon, axis=1)
    area_flat = area2d.reshape(-1)
    tgt_doy = d0[4]

    # windowed definitions sweep the +/-day window; seasonal ones (EURO-CORDEX) have
    # a single window-independent threshold -> one curve.
    windows_for = lambda d: [0] if d.kind == "season" else WINDOWS

    # detect for every (definition, window)
    res = {}
    for d in defs:
        (ref_stats, ref_doy, ref_months, tgt_stats, td, tgt_months,
         _, _) = data[(d.tag, d.ref_years)]
        for w in windows_for(d):
            hot = DET.hot_mask(d, ref_stats, ref_doy, tgt_stats, td, w,
                               ref_months=ref_months, tgt_months=tgt_months, lat=latc)
            dur, ar = DET.spell_events(hot, area_flat)
            daily = (DET.active_mask(hot) * area2d[None]).sum(axis=(1, 2)) / 1e6
            res[(d.name, w)] = (dur, ar, daily)
            wlab = "seasonal" if d.kind == "season" else f"+/-{w}d"
            print(f"[compare] {d.name} {wlab}: {dur.size} events, "
                  f"{ar.sum()/1e6:.1f} x10^6 km^2", flush=True)

    rtag = "" if REGION == "world" else f"_{REGION}"
    rlab = "" if REGION == "world" else f"{REGION.replace('_', ' ').title()} — "
    def_leg = [Line2D([], [], color=D.COLORS[d.name], lw=2.4,
                      label=DEF_LABELS.get(d.name, d.name)) for d in defs]
    win_leg = [Line2D([], [], color=shades("#3a3a3a", len(WINDOWS))[i], lw=2.4,
                      label=f"$\\pm${w}") for i, w in enumerate(WINDOWS)]

    # ---- 1. amount vs duration (count + area) ----
    dmax = max((dur.max() if dur.size else 3) for dur, _, _ in res.values())
    bins = np.arange(3, dmax + 1)
    for metric, ylabel in [("count", "Number of heat waves (grid cells)"),
                           ("area", "Area affected (10$^6$ km$^2$)")]:
        fig, ax = plt.subplots(figsize=(7.8, 4.9))
        for d in defs:
            seasonal = d.kind == "season"
            cols = None if seasonal else shades(D.COLORS[d.name], len(WINDOWS))
            for i, w in enumerate(windows_for(d)):
                dur, ar, _ = res[(d.name, w)]
                y = (np.array([(dur == b).sum() for b in bins]) if metric == "count"
                     else np.array([ar[dur == b].sum() / 1e6 for b in bins]))
                ax.plot(bins, y, color=(D.COLORS[d.name] if seasonal else cols[i]),
                        lw=2.2 if seasonal else 1.5)
        ax.set_yscale("log")
        ax.set_title(f"Length of heat wave classification in {REGION_NAME} in "
                     f"{YEAR} by definitions (> {D.PTAG})",
                     fontsize=12.5, color=INK, loc="left")
        ax.set_xlabel("Heat-wave duration (consecutive days)", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.grid(True, which="both", color=GRID, lw=0.6)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        lwin = ax.legend(handles=win_leg, title="time window", loc="upper right",
                         bbox_to_anchor=(1.0, 1.0), ncol=len(WINDOWS),
                         fontsize=8, title_fontsize=8, frameon=False)
        ax.add_artist(lwin)
        ax.legend(handles=def_leg, title="definition", loc="upper right",
                  bbox_to_anchor=(1.0, 0.80), fontsize=8, title_fontsize=8, frameon=False)
        fig.tight_layout()
        out = FIGDIR / f"heatwave{YEAR}_defcompare_{metric}_{D.PTAG}_wsweep_2p8deg{rtag}.pdf"
        fig.savefig(out, bbox_inches="tight"); plt.close(fig)
        print(f"[compare] wrote {out.name}", flush=True)

    # ---- 2. timing over the year (area in heat wave per day) ----
    tp, tl = month_ticks(YEAR)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for d in defs:
        seasonal = d.kind == "season"
        cols = None if seasonal else shades(D.COLORS[d.name], len(WINDOWS))
        for i, w in enumerate(windows_for(d)):
            _, _, daily = res[(d.name, w)]
            ax.plot(tgt_doy, daily, color=(D.COLORS[d.name] if seasonal else cols[i]),
                    lw=1.7 if seasonal else 1.1)
    ax.set_xticks(tp); ax.set_xticklabels(tl)
    ax.set_xlim(1, tgt_doy.max())
    ax.set_title(f"Area classified as heat wave in {REGION_NAME} in {YEAR} "
                 f"by definitions (> {D.PTAG})", fontsize=13, color=INK, loc="left")
    ax.set_xlabel("Month", color=INK)
    ax.set_ylabel("Area in heat wave (10$^6$ km$^2$)", color=INK)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    l1 = ax.legend(handles=def_leg, title="definition", loc="upper left",
                   fontsize=8, title_fontsize=8, frameon=False)
    ax.add_artist(l1)
    ax.legend(handles=win_leg, title="time window", loc="upper right",
              fontsize=8, title_fontsize=8, ncol=len(WINDOWS), frameon=False)
    fig.tight_layout()
    out = FIGDIR / f"heatwave{YEAR}_defcompare_timing_{D.PTAG}_wsweep_2p8deg{rtag}.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[compare] wrote {out.name}", flush=True)
    print(f"[compare] DONE -> {FIGDIR}", flush=True)


if __name__ == "__main__":
    main()
