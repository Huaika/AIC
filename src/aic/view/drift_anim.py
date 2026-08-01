#!/usr/bin/env python
"""Per-day day-10 drift-error maps animated over a month (GIF), Europe.

Each frame is one day v of the month: the day-10 NeuralGCM forecast field valid
on v (from the rollout initialized v-10 days earlier) MINUS the ERA5 truth on v
-- i.e. the day-10 forecast error map for that day. The colour scale is held
FIXED across the whole month (computed in a first pass). GIF at 2 fps (0.5 s per
frame). All frames share ONE palette (aic.view.gif_utils) so the colouring is
consistent across the animation (no per-frame GIF-palette flicker).

Variables / levels: temperature @ 850 hPa, geopotential @ 500 hPa, and wind speed
sqrt(u^2 + v^2) @ 850 hPa (u,v combined per frame). Region: europe only.

Run via EVAL_RUN + EVAL_MONTHS (comma list of months 1..12). Output:
outputs/figures/<run>/<0m_month>/europe/<variable>/drift_maps/<name>.gif
"""
from __future__ import annotations

import calendar
import io

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from aic.controller.eval import eval_common as C
from aic.view import naming as fig_naming
from aic.view.gif_utils import shared_palette, quantize, save_gif

YEAR = C.YEAR
FINAL_LEAD_H = 240          # day-10
LEAD_DAYS = 10
FPS = 2                     # 0.5 s per frame
REGION = "europe"

# (variable label, level hPa, kind)   kind: "scalar" | "wind"
SPECS = [
    ("temperature", 850, "scalar"),
    ("geopotential", 500, "scalar"),
    ("wind_speed", 850, "wind"),
]
UNITS = {"temperature": "K", "geopotential": "m^2/s^2", "wind_speed": "m/s"}
# field colormap for the forecast + truth panels (difference panel is always RdBu_r)
FIELD_CMAP = {"temperature": "RdYlBu_r", "geopotential": "viridis",
              "wind_speed": "viridis"}


def _pred_path(init_date: str):
    return C.PRED_DIR / f"pred_{YEAR}_{init_date}.nc"


def _truth_field(kind: str, var: str, level: int, valid_time) -> xr.DataArray:
    """ERA5 truth on the model grid at ``valid_time`` (nearest)."""
    if kind == "wind":
        tu = C.truth_at_levels("u_component_of_wind", [level]).sel(level=level)
        tv = C.truth_at_levels("v_component_of_wind", [level]).sel(level=level)
        speed = np.sqrt(tu ** 2 + tv ** 2)
        return speed.sel(time=valid_time, method="nearest")
    t = C.truth_at_levels(var, [level]).sel(level=level)
    return t.sel(time=valid_time, method="nearest")


def _daily_fields(var: str, level: int, kind: str, month: int):
    """Per day of the month: (date, forecast, truth, error) fields, europe-cropped
    (lat, lon). forecast = NeuralGCM day-10 field valid that day; truth = ERA5 that
    day; error = forecast - truth."""
    days = [pd.Timestamp(YEAR, month, d)
            for d in range(1, calendar.monthrange(YEAR, month)[1] + 1)]
    out = []
    for v in days:
        init = (v - pd.Timedelta(days=LEAD_DAYS)).date()
        p = _pred_path(str(init))
        if not p.exists():
            continue
        ds = xr.open_dataset(p)
        end = ds.isel(time=-1)
        if kind == "wind":
            uu = end["u_component_of_wind"].sel(level=level)
            vv = end["v_component_of_wind"].sel(level=level)
            fc = np.sqrt(uu ** 2 + vv ** 2)
        else:
            fc = end[var].sel(level=level)
        valid = ds["valid_time"].isel(time=-1).values
        tr = _truth_field(kind, var, level, valid)
        crop = lambda d: C.select_region(d, REGION).transpose("latitude", "longitude")
        fc_c, tr_c = crop(fc), crop(tr)
        out.append((v, fc_c.compute(), tr_c.compute(), (fc_c - tr_c).compute()))
        ds.close()
    return out


def _animate(var: str, level: int, kind: str, month: int):
    frames = _daily_fields(var, level, kind, month)
    if not frames:
        print(f"  [skip] {var}@{level} {C.period_dir_name(month)}: no frames")
        return
    # Fixed scales across the month: field range (forecast+truth) and diverging
    # error limit -- so every frame is directly comparable.
    fstack = np.stack([f.values for _, f, _, _ in frames]
                      + [t.values for _, _, t, _ in frames])
    vmin, vmax = float(np.nanmin(fstack)), float(np.nanmax(fstack))
    dlim = float(np.nanpercentile(np.abs(np.stack([d.values for *_, d in frames])), 99)) or 1.0
    lon = frames[0][1].longitude.values
    lat = frames[0][1].latitude.values
    w, e, s, n = C.region_extent(REGION)
    units = UNITS[var]
    label = var.replace("_", " ")
    fcmap = FIELD_CMAP.get(var, "viridis")

    images = []
    for d, fc, tr, dr in frames:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
        for ax, fld, ttl, cm, lo, hi, cl in (
            (axes[0], fc, "NeuralGCM day-10 forecast", fcmap, vmin, vmax, f"{label} [{units}]"),
            (axes[1], tr, f"{C.REF_LABEL} truth",       fcmap, vmin, vmax, f"{label} [{units}]"),
            (axes[2], dr, "day-10 error (forecast - truth)", "RdBu_r", -dlim, dlim, f"error [{units}]"),
        ):
            m = ax.pcolormesh(lon, lat, fld, cmap=cm, vmin=lo, vmax=hi, shading="auto")
            C.draw_coastlines(ax)
            ax.set_xlim(w, e); ax.set_ylim(s, n)
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title(ttl)
            fig.colorbar(m, ax=ax, shrink=0.8, label=cl)
        fig.suptitle(f"{C.REF_LABEL} — {label}@{level} hPa, day-10 rollout  |  "
                     f"valid {pd.Timestamp(d).date()} "
                     f"(init {(pd.Timestamp(d)-pd.Timedelta(days=LEAD_DAYS)).date()})",
                     fontsize=13)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches=None)  # fixed size
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))

    figdir = C.figure_dir(month, REGION, var, "drift_maps")
    out = figdir / fig_naming.figure_name(
        C.MODEL, C.DATASET, REGION, var, level, C.YEAR,
        fig_naming.months_token(month), "drift-map-anim", ext="gif")
    # one shared palette for the whole animation -> consistent colouring, no flicker
    pal = shared_palette(images)
    save_gif([quantize(im, pal) for im in images], out, duration=int(1000 / FPS))
    print(f"  saved {out.name} ({len(frames)} frames, field {vmin:.4g}..{vmax:.4g}, "
          f"dlim={dlim:.4g} {units})")


def main():
    periods = [p for p in C.selected_periods() if p != 0]
    if not periods:
        raise SystemExit("set EVAL_MONTHS to the month(s) 1..12 to animate")
    for month in periods:
        print(f"=== {C.period_dir_name(month)} ({REGION}) ===")
        for var, level, kind in SPECS:
            _animate(var, level, kind, month)
    print(f"done -> {C.FIGROOT}/<month>/{REGION}/<variable>/drift_maps/*.gif")


if __name__ == "__main__":
    main()
