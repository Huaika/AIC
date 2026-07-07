#!/usr/bin/env python
"""Per-day day-10 drift-error maps animated over a month (GIF), Europe.

Each frame is one day v of the month: the day-10 NeuralGCM forecast field valid
on v (from the rollout initialized v-10 days earlier) MINUS the ERA5 truth on v
-- i.e. the day-10 forecast error map for that day. The colour scale is held
FIXED across the whole month (computed in a first pass). GIF at 2 fps (0.5 s per
frame).

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

from ngcm import eval_common as C
from common import fig_naming

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


def _daily_drifts(var: str, level: int, kind: str, month: int):
    """List of (date, europe-cropped day-10 error field) for every day in month."""
    days = [pd.Timestamp(YEAR, month, d)
            for d in range(1, calendar.monthrange(YEAR, month)[1] + 1)]
    fc_var = "wind_speed" if kind == "wind" else var
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
        drift = C.select_region(fc - tr, REGION).transpose("latitude", "longitude")
        out.append((v, drift.compute()))
        ds.close()
    return out


def _animate(var: str, level: int, kind: str, month: int):
    frames = _daily_drifts(var, level, kind, month)
    if not frames:
        print(f"  [skip] {var}@{level} {C.period_dir_name(month)}: no frames")
        return
    stack = np.stack([f.values for _, f in frames])
    dlim = float(np.nanpercentile(np.abs(stack), 99)) or 1.0
    lon = frames[0][1].longitude.values
    lat = frames[0][1].latitude.values
    w, e, s, n = C.region_extent(REGION)
    units = UNITS[var]
    label = var.replace("_", " ")

    # Render each day to a fixed-size PNG, then assemble one GIF (0.5 s/frame,
    # loop forever). Color scale (dlim) is fixed across all frames.
    images = []
    for d, fld in frames:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        m = ax.pcolormesh(lon, lat, fld, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                          shading="auto")
        C.draw_coastlines(ax)
        ax.set_xlim(w, e); ax.set_ylim(s, n)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_title(f"{C.REF_LABEL} — day-10 {label}@{level}hPa error\n"
                     f"valid {pd.Timestamp(d).date()}  "
                     f"(init {(pd.Timestamp(d)-pd.Timedelta(days=LEAD_DAYS)).date()})")
        fig.colorbar(m, ax=ax, shrink=0.85, label=f"day-10 error [{units}]")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)   # fixed size (no tight bbox)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))

    figdir = C.figure_dir(month, REGION, var, "drift_maps")
    out = figdir / fig_naming.figure_name(
        C.MODEL, C.DATASET, REGION, var, level, C.YEAR,
        fig_naming.months_token(month), "drift-map-anim", ext="gif")
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=int(1000 / FPS), loop=0)
    print(f"  saved {out.name} ({len(frames)} frames, dlim={dlim:.4g} {units})")


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
