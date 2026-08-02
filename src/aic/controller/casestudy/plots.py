#!/usr/bin/env python
"""Case study: NeuralGCM rollouts over EUROPE heat-wave episodes (mixture / p99).

For each detected episode (see ``heatwave_mask``) this renders, using ONLY the grid
cells in the episode's heat-wave FOOTPRINT (a ``GridPoints`` set):

  Lagrangian (follow the afflicted region over its heat-wave lifetime + rollout
  window before + brief window after):
    * spaghetti   -- footprint-area-mean field, ERA5 (thick black) + one thin line
                     per rollout init in [start-10d, end], over valid [start-10d,
                     end+10d]; the episode span is shaded.
    * rmse-vs-lead-- RMSE + bias vs lead over the footprint, aggregated over the
                     rollouts initialised in [start-10d, end] (the forecasts rolling
                     into the event).

  Eulerian (the whole footprint analysed over the episode's day-10 valid window;
  cells never in the heat wave are left NaN / uncoloured):
    * drift map   -- ERA5 mean | forecast day-10 mean | mean day-10 error, cropped
                     to Europe, coloured only on the footprint.

Variables: temperature@850 hPa and geopotential@500 hPa (env EVAL_VARS / CS_LEVELS
override). The definition + percentile come from ``HW_PCT`` (=0.99) + the mixture
definition. Figures: ``outputs/figures/case_study/<def>_<ptag>/<year>/<episode>/``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from aic.controller.eval import eval_common as C
from aic.controller.eval import gridpoints as GP
from aic.controller.eval import sources as S
from aic.view import drift as drift_view
from aic.controller.casestudy import heatwave_mask as HM
from aic.controller.heatwave import definitions as D
from aic.regions import region_extent

BEFORE = int(os.environ.get("CS_BEFORE", "10"))    # rollout window before episode
AFTER = int(os.environ.get("CS_AFTER", "10"))      # brief window after (valid axis)
FINAL_LEAD_H = S.FINAL_DAY_LEAD_MIN                 # >=216 h == day-10 slab
# variable -> level(s) to evaluate (T at 850, Z at 500 for the case study)
CS_LEVELS = {"temperature": [850], "geopotential": [500]}
INK = "#222222"; GRID = "#dddddd"


# --------------------------------------------------------------------------- #
# episode -> grid-point set + rollout file window
# --------------------------------------------------------------------------- #
def footprint_points(source: S.Source, ep: HM.Episode, year: int) -> GP.GridPoints:
    """The episode's footprint as a GridPoints on the source's prediction grid
    (coords reassigned to the pred grid so it aligns with rollouts/truth exactly)."""
    lat, lon = source.prediction_grid()
    m = xr.DataArray(ep.footprint.values, dims=("latitude", "longitude"),
                     coords={"latitude": lat, "longitude": lon})
    return GP.GridPoints.from_mask(f"hw_{year}_{ep.tag}", m, region_extent("europe"))


def episode_files(source: S.Source, ep: HM.Episode, before: int, after: int):
    """Rollout pred files whose INIT date lies in [start-before, end+after]."""
    lo = (ep.start - pd.Timedelta(days=before)).date()
    hi = (ep.end + pd.Timedelta(days=after)).date()
    out = []
    for f in source.pred_files():
        d = pd.to_datetime(f.stem.replace(f"pred_{source.year}_", "")).date()
        if lo <= d <= hi:
            out.append(f)
    return out


def fig_dir(defn, year: int, ep: HM.Episode) -> Path:
    d = (C.FIG_ROOT / "case_study" / f"{defn.name}_{D.PTAG}" / str(year)
         / f"{ep.tag}_{ep.start:%Y-%m-%d}_{ep.end:%Y-%m-%d}")
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Lagrangian: spaghetti over the episode window (footprint area-mean)
# --------------------------------------------------------------------------- #
def spaghetti_episode(source, defn, year, ep, gp, var, lev):
    meta = C.VARIABLES[var]
    label, units = meta["label"], meta["units"]
    v0 = ep.start - pd.Timedelta(days=BEFORE)
    v1 = ep.end + pd.Timedelta(days=AFTER)
    files = episode_files(source, ep, BEFORE, 0)     # inits rolling into the event
    if not files:
        print(f"  [spaghetti] {ep.tag} {var}: no rollouts in window; skip")
        return
    roll = source.rollout_gmean(var, meta["short"], [lev], [gp],
                                files=files, cache=False)

    # ERA5 reference: footprint-area-mean, daily, over the valid window
    truth = source.truth_at_levels(var, [lev])
    ref = GP.masked_area_mean(truth.sel(level=lev), gp)
    ref = ref.to_dataframe(name="ref").reset_index()
    ref["date"] = pd.to_datetime(ref["time"]).dt.floor("D")
    ref = ref.groupby("date", as_index=False)["ref"].mean()
    ref = ref[(ref["date"] >= v0) & (ref["date"] <= v1)]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axvspan(ep.start, ep.end, color="#f2c14e", alpha=0.25, zorder=0,
               label="heat-wave episode")
    r = roll.copy()
    r["valid_time"] = r["init_date"] + pd.to_timedelta(r["lead_hours"], unit="h")
    r["lead_day_idx"] = (r["lead_hours"] // 24).astype(int)
    for d in sorted(r["init_date"].unique()):
        g = r[r["init_date"] == d]
        daily = (g.groupby("lead_day_idx")
                   .agg(vt=("valid_time", "mean"), val=("pred_gmean", "mean"))
                   .reset_index())
        ax.plot(daily["vt"], daily["val"], color=source.color, lw=0.7, alpha=0.55,
                zorder=2)
    ax.plot([], [], color=source.color, lw=1.6, label=f"{source.pretty} 10-day rollout")
    ax.plot(ref["date"], ref["ref"], color="black", lw=2.2, zorder=3,
            label=f"{source.ref_label} (daily mean)")
    ax.set_title(f"Europe heat-wave {ep.label} — footprint-mean {label} @ {lev} hPa "
                 f"({defn.name}, > {D.PTAG})", fontsize=12, color=INK, loc="left")
    ax.set_ylabel(f"{label} @{lev}hPa footprint mean [{units}]", color=INK)
    ax.set_xlabel("Valid time", color=INK)
    ax.set_xlim(v0, v1)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.25); ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = fig_dir(defn, year, ep) / f"spaghetti_{meta['short']}_L{lev:04d}.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out.name}")


# --------------------------------------------------------------------------- #
# Lagrangian: RMSE + bias vs lead over the footprint (episode rollouts)
# --------------------------------------------------------------------------- #
def rmse_episode(source, defn, year, ep, gp, var, lev):
    meta = C.VARIABLES[var]
    label, units = meta["label"], meta["units"]
    files = episode_files(source, ep, BEFORE, 0)
    if not files:
        print(f"  [rmse] {ep.tag} {var}: no rollouts in window; skip")
        return
    per = source.drift_per_init(var, meta["short"], [lev], [gp],
                                files=files, cache=False)
    agg = drift_view.aggregate(per)
    a = agg[(agg["region"] == gp.key) & (agg["level"] == lev)].sort_values("lead_hours")
    if a.empty:
        print(f"  [rmse] {ep.tag} {var}: empty aggregate; skip")
        return
    fig, axr = plt.subplots(figsize=(6.6, 4.4))
    axb = axr.twinx()
    axr.plot(a["lead_day"], a["rmse"], color="#1f77b4", lw=1.9, label="RMSE")
    axb.plot(a["lead_day"], a["bias"], color="#d62728", lw=1.5, label="bias")
    axb.axhline(0.0, color="0.4", lw=0.8, ls=":", alpha=0.6)
    axr.set_title(f"Europe heat-wave {ep.label} — footprint {label}@{lev} hPa "
                  f"(mean of {int(a['n_init'].iloc[0])} rollouts)",
                  fontsize=11, color=INK, loc="left")
    axr.set_xlabel("lead time (days)")
    axr.set_ylabel(f"RMSE [{units}]", color="#1f77b4")
    axb.set_ylabel(f"mean bias [{units}]", color="#d62728")
    axr.tick_params(axis="y", labelcolor="#1f77b4")
    axb.tick_params(axis="y", labelcolor="#d62728")
    axr.grid(True, alpha=0.3)
    fig.tight_layout()
    out = fig_dir(defn, year, ep) / f"rmse-vs-lead_{meta['short']}_L{lev:04d}.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out.name}")


# --------------------------------------------------------------------------- #
# Eulerian: mean day-10 error map over the footprint (Europe extent)
# --------------------------------------------------------------------------- #
def driftmap_episode(source, defn, year, ep, gp, var, lev):
    meta = C.VARIABLES[var]
    label, units, fcmap = meta["label"], meta["units"], meta["cmap"]
    files = episode_files(source, ep, BEFORE, 0)
    if not files:
        print(f"  [driftmap] {ep.tag} {var}: no rollouts in window; skip")
        return
    truth = source.truth_at_levels(var, [lev]).sel(level=lev)
    facc = None; racc = None; n = 0
    for f in files:
        ds = xr.open_dataset(f)
        hot10 = ds["lead_hours"] >= FINAL_LEAD_H
        day10 = (ds[var].sel(level=lev).where(hot10, drop=True)
                 .transpose("time", "latitude", "longitude"))
        if day10.sizes["time"] == 0:
            ds.close(); continue
        vt = ds["valid_time"].where(hot10, drop=True).values
        tru = (truth.sel(time=vt, method="nearest")
               .transpose("time", "latitude", "longitude"))
        fsum = day10.sum("time")
        rsum = tru.sum("time").assign_coords(latitude=fsum.latitude,
                                             longitude=fsum.longitude)
        facc = fsum if facc is None else facc + fsum
        racc = rsum if racc is None else racc + rsum
        n += day10.sizes["time"]
        ds.close()
    if not n:
        print(f"  [driftmap] {ep.tag} {var}: no day-10 steps; skip")
        return
    fc = facc / n
    rf = racc / n
    dr = fc - rf
    fpm = gp.mask
    fc = fc.where(fpm); rf = rf.where(fpm); dr = dr.where(fpm)

    w, e, s, n_ = region_extent("europe")
    gm = float(GP.masked_area_mean(dr, gp))
    dlim = float(np.nanpercentile(np.abs(dr.values), 99)) or 1.0
    vmin = float(np.nanmin([np.nanmin(fc.values), np.nanmin(rf.values)]))
    vmax = float(np.nanmax([np.nanmax(fc.values), np.nanmax(rf.values)]))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for ax, field, cmap, vlo, vhi, ttl, clab in [
        (axes[0], rf, fcmap, vmin, vmax, f"{source.ref_label} mean", f"{label} [{units}]"),
        (axes[1], fc, fcmap, vmin, vmax, f"{source.pretty} day-10 mean", f"{label} [{units}]"),
        (axes[2], dr, "RdBu_r", -dlim, dlim,
         f"mean day-10 error\n(footprint mean {gm:+.4g} {units})", f"error [{units}]")]:
        m = ax.pcolormesh(field.longitude, field.latitude, field, cmap=cmap,
                          vmin=vlo, vmax=vhi, shading="auto")
        fig.colorbar(m, ax=ax, shrink=0.82, label=clab)
        C.draw_coastlines(ax)
        ax.set_xlim(w, e); ax.set_ylim(s, n_)
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.2)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.suptitle(f"Europe heat-wave {ep.label} — day-10 {label}@{lev} hPa over the "
                 f"heat-wave footprint ({ep.n_cells} cells, {defn.name} > {D.PTAG})",
                 y=1.03, fontsize=12.5)
    fig.tight_layout()
    out = fig_dir(defn, year, ep) / f"drift-map_{meta['short']}_L{lev:04d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out.name}")


# --------------------------------------------------------------------------- #
# per-year overview: coverage timeline with episodes shaded
# --------------------------------------------------------------------------- #
def year_overview(defn, year, active_da, eps, region="europe"):
    cov = HM.region_coverage(active_da, region)
    times = pd.to_datetime(active_da["time"].values)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(times, 100 * cov, color="#b0413e", lw=1.3)
    ax.fill_between(times, 0, 100 * cov, color="#d98880", alpha=0.35)
    for e in eps:
        ax.axvspan(e.start, e.end, color="#f2c14e", alpha=0.3, zorder=0)
        ax.text(e.start, ax.get_ylim()[1] * 0.92, e.tag, fontsize=7, color="#666")
    ax.set_title(f"Europe heat-wave coverage in {year} ({defn.name} > {D.PTAG}, "
                 f"{len(eps)} episodes)", fontsize=12, color=INK, loc="left")
    ax.set_ylabel("% of Europe in heat wave", color=INK)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(times[0], times[-1]); ax.margins(x=0)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    d = C.FIG_ROOT / "case_study" / f"{defn.name}_{D.PTAG}" / str(year)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"_overview_coverage_{year}.pdf"
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[overview] wrote {out}")


# --------------------------------------------------------------------------- #
def run_year(source, defn, year, window=HM.DEFAULT_WINDOW, region="europe"):
    print(f"=== case study {defn.name} > {D.PTAG}: {source.run} ({year}) ===",
          flush=True)
    active = HM.active_mask_da(defn, year, window, region)
    eps = HM.episodes(active, region)
    only = os.environ.get("CS_ONLY_EP", "").strip()
    if only:
        want = {int(x) for x in only.replace(",", " ").split()}
        eps = [e for e in eps if e.idx in want]
    print(f"[{year}] {len(eps)} episode(s): "
          + ", ".join(f"{e.tag}({e.label},{e.n_cells}c)" for e in eps), flush=True)
    year_overview(defn, year, active, eps, region)
    variables = C.selected_variables()
    for ep in eps:
        gp = footprint_points(source, ep, year)
        print(f"-- {ep.tag} {ep.label} ({ep.n_days}d, {ep.n_cells} cells) --", flush=True)
        for var in variables:
            for lev in CS_LEVELS.get(var, [850]):
                if lev not in source.prediction_levels():
                    continue
                spaghetti_episode(source, defn, year, ep, gp, var, lev)
                rmse_episode(source, defn, year, ep, gp, var, lev)
                driftmap_episode(source, defn, year, ep, gp, var, lev)


def main():
    defn = D.BY_NAME[os.environ.get("HW_CS_DEF", "mixture")]
    year = int(os.environ.get("HW_YEAR", "2023"))
    run = os.environ.get("EVAL_RUN", f"era5_{year}")
    source = S.Source.from_run(run)
    run_year(source, defn, year)
    print(f"done -> {C.FIG_ROOT}/case_study/{defn.name}_{D.PTAG}/{year}/")


if __name__ == "__main__":
    main()
