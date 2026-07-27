#!/usr/bin/env python
"""Per-day day-10 drift-error maps animated over a month (GIF), Europe only.

Each frame is one day ``v`` of the month:

* forecast = final field of the rollout initialized ``v - 10`` days earlier,
* truth    = ERA5 field on ``v`` (nearest-time),
* error    = forecast - truth.

The color scale is held FIXED across the whole month (computed in a first pass)
so frames are directly comparable. GIF at 2 fps (0.5 s per frame).

Variables / levels: temperature @ 850 hPa, geopotential @ 500 hPa, wind speed
``sqrt(u^2 + v^2)`` @ 850 hPa (u,v combined per frame). Region: europe only.
Run via ``EVAL_RUN + EVAL_MONTHS`` (comma list of months 1..12). Output:
``outputs/figures/<run>/<0m_month>/europe/<variable>/drift_maps/<name>.gif``.
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

from ngcm import eval_common as C
from ngcm.plots import _helpers as H
from common import fig_naming

YEAR = C.YEAR
FINAL_LEAD_H = 240  # day-10
LEAD_DAYS = 10
FPS = 2  # 0.5 s per frame
REGION = "europe"

# (variable label, level hPa, kind)   kind: "scalar" | "wind"
SPECS = [
    ("temperature", 850, "scalar"),
    ("geopotential", 500, "scalar"),
    ("wind_speed", 850, "wind"),
]
UNITS = {"temperature": "K", "geopotential": "m^2/s^2", "wind_speed": "m/s"}
# Field colormap for the forecast + truth panels (diff panel is always RdBu_r).
FIELD_CMAP = {"temperature": "RdYlBu_r", "geopotential": "viridis",
              "wind_speed": "viridis"}


def _pred_path(init_date: str):
    return C.PRED_DIR / f"pred_{YEAR}_{init_date}.nc"


def _crop(da: xr.DataArray) -> xr.DataArray:
    """Europe crop with a canonical (lat, lon) transpose."""
    return C.select_region(da, REGION).transpose("latitude", "longitude")


def _truth_field(kind: str, var: str, level: int, valid_time) -> xr.DataArray:
    """ERA5 truth on the model grid at ``valid_time`` (nearest)."""
    if kind == "wind":
        tu = C.truth_at_levels("u_component_of_wind", [level]).sel(level=level)
        tv = C.truth_at_levels("v_component_of_wind", [level]).sel(level=level)
        speed = np.sqrt(tu ** 2 + tv ** 2)
        return speed.sel(time=valid_time, method="nearest")
    t = C.truth_at_levels(var, [level]).sel(level=level)
    return t.sel(time=valid_time, method="nearest")


def _forecast_field(ds: xr.Dataset, kind: str, var: str, level: int) -> xr.DataArray:
    """Final (day-10) forecast field on the model grid."""
    end = ds.isel(time=-1)
    if kind == "wind":
        uu = end["u_component_of_wind"].sel(level=level)
        vv = end["v_component_of_wind"].sel(level=level)
        return np.sqrt(uu ** 2 + vv ** 2)
    return end[var].sel(level=level)


def _daily_fields(var: str, level: int, kind: str, month: int):
    """One tuple per day of ``month``: ``(date, forecast, truth, error)``,
    europe-cropped to (lat, lon)."""
    days = [pd.Timestamp(YEAR, month, d)
            for d in range(1, calendar.monthrange(YEAR, month)[1] + 1)]
    out = []
    for v in days:
        init = (v - pd.Timedelta(days=LEAD_DAYS)).date()
        p = _pred_path(str(init))
        if not p.exists():
            continue
        ds = xr.open_dataset(p)
        fc = _forecast_field(ds, kind, var, level)
        valid = ds["valid_time"].isel(time=-1).values
        tr = _truth_field(kind, var, level, valid)
        fc_c, tr_c = _crop(fc), _crop(tr)
        out.append((v, fc_c.compute(), tr_c.compute(), (fc_c - tr_c).compute()))
        ds.close()
    return out


def _frame_scales(frames):
    """Shared color scales: field range from forecast+truth, symmetric diff limit."""
    fstack = np.stack([f.values for _, f, _, _ in frames]
                      + [t.values for _, _, t, _ in frames])
    vmin, vmax = float(np.nanmin(fstack)), float(np.nanmax(fstack))
    dstack = np.stack([d.values for *_, d in frames])
    dlim = float(np.nanpercentile(np.abs(dstack), 99)) or 1.0
    return vmin, vmax, dlim


def _render_frame(d, fc, tr, dr, *, lon, lat, extent, vmin, vmax, dlim,
                  var, level, units, label, fcmap) -> Image.Image:
    """One PNG frame as a PIL image."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    titles = (
        "NeuralGCM day-10 forecast",
        f"{C.REF_LABEL} truth",
        "day-10 error (forecast - truth)",
    )
    H.plot_three_panel_maps(
        axes, fig, lon, lat, fc, tr, dr,
        field_cmap=fcmap, units=units, field_label=label,
        diff_label="error", titles=titles, extent=extent,
        vmin=vmin, vmax=vmax, dlim=dlim,
    )
    fig.suptitle(f"{C.REF_LABEL} — {label}@{level} hPa, day-10 rollout  |  "
                 f"valid {pd.Timestamp(d).date()} "
                 f"(init {(pd.Timestamp(d) - pd.Timedelta(days=LEAD_DAYS)).date()})",
                 fontsize=13)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches=None)  # fixed size
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _animate(var: str, level: int, kind: str, month: int):
    frames = _daily_fields(var, level, kind, month)
    if not frames:
        print(f"  [skip] {var}@{level} {C.period_dir_name(month)}: no frames")
        return
    vmin, vmax, dlim = _frame_scales(frames)

    lon = frames[0][1].longitude.values
    lat = frames[0][1].latitude.values
    extent = C.region_extent(REGION)
    units = UNITS[var]
    label = var.replace("_", " ")
    fcmap = FIELD_CMAP.get(var, "viridis")

    images = [
        _render_frame(d, fc, tr, dr,
                      lon=lon, lat=lat, extent=extent,
                      vmin=vmin, vmax=vmax, dlim=dlim,
                      var=var, level=level, units=units,
                      label=label, fcmap=fcmap)
        for d, fc, tr, dr in frames
    ]

    figdir = C.figure_dir(month, REGION, var, "drift_maps")
    out = figdir / fig_naming.figure_name(
        C.MODEL, C.DATASET, REGION, var, level, C.YEAR,
        fig_naming.months_token(month), "drift-map-anim", ext="gif")
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=int(1000 / FPS), loop=0)
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
