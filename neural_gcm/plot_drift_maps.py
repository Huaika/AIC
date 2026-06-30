#!/usr/bin/env python
"""10-day drift maps (Rackow et al. 2024 Fig. 3 style) -- RUN/VARIABLE/REGION-agnostic.

drift = annual-mean of the model's day-10 fields minus the reference annual mean:
  ngcm_day10_clim = mean over inits of the rollout's final-day mean (216..240 h)
  ref_clim        = reference annual-mean field
  drift           = ngcm_day10_clim - ref_clim   (model grid)
Three panels per (region, variable, level): NeuralGCM day-10 clim, reference, drift.

The clim/drift fields are GLOBAL and cached once per (run, variable); each region
is just that global field CROPPED to the region's box (no recompute). Run via
EVAL_RUN, variables via EVAL_VARS, regions via EVAL_REGIONS. Output:
figures/<run>/<region>/<variable>/drift_maps/.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import eval_common as C

YEAR = C.YEAR
FINAL_DAY_LEAD_MIN = 216


def build_fields(var, short, levels, truth, period) -> xr.Dataset | None:
    """Day-10 clim / reference clim / drift fields for a period, cached per
    (run, var, period). period 0 = entire year (reuses the existing annual cache
    name); month m = mean over that month's init-days only (truth restricted to
    month m). Returns None if the period has no init-days."""
    suffix = "" if period == 0 else f"_{period:02d}"
    nc = C.OUTDIR / f"{C.RUN}_drift_maps_{short}{suffix}_{C.level_tag()}.nc"
    if nc.exists():
        print(f"[maps] cached {nc}")
        return xr.open_dataset(nc)

    files = sorted(C.PRED_DIR.glob(f"pred_{YEAR}_*.nc"))
    if period == 0:
        ref_clim = truth.mean("time")
    else:
        files = [f for f in files if C.pred_init_month(f) == period]
        mask = (truth["time"].dt.month == period).values
        ref_clim = truth.isel(time=mask).mean("time")
    if not files:
        print(f"[maps] no init-days for {C.period_dir_name(period)}; skip")
        return None
    ref_clim = ref_clim.transpose("level", "latitude", "longitude")
    print(f"[maps] {C.period_dir_name(period)}: end-of-forecast mean over "
          f"{len(files)} rollouts ({var}), {len(levels)} levels")
    acc, n = None, 0
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        t = ds[var].sel(level=levels)
        end = t.where(ds["lead_hours"] >= FINAL_DAY_LEAD_MIN, drop=True).mean("time")
        end = end.transpose("level", "latitude", "longitude")
        acc = end if acc is None else acc + end
        n += 1
        ds.close()
        if i % 50 == 0 or i == len(files) - 1:
            print(f"  {i + 1}/{len(files)}")
    ngcm = acc / n
    out = xr.Dataset({"ngcm_day10_clim": ngcm, "ref_annual_clim": ref_clim,
                      "drift": ngcm - ref_clim})
    out.attrs.update(n_inits=n, final_day_lead_min_h=FINAL_DAY_LEAD_MIN,
                     variable=var, period=C.period_dir_name(period),
                     drift_def=f"mean(end-of-10day-forecast) - {C.REF_LABEL} mean")
    out.to_netcdf(nc)
    print(f"[maps] wrote {nc}")
    return out


def plot_period_region(out, var, short, units, label, fcmap, levels, period, reg):
    figdir = C.figure_dir(period, reg, var, "drift_maps")
    w, e, s, n_ = C.region_extent(reg)
    area = "global" if reg == "world" else reg
    for lev in levels:
        ng = C.select_region(out["ngcm_day10_clim"].sel(level=lev), reg)
        rf = C.select_region(out["ref_annual_clim"].sel(level=lev), reg)
        dr = C.select_region(out["drift"].sel(level=lev), reg)
        lon, lat = ng.longitude, ng.latitude
        vmin = float(min(ng.min(), rf.min())); vmax = float(max(ng.max(), rf.max()))
        dlim = float(np.nanpercentile(np.abs(dr.values), 99)) or 1.0

        fig, axes = plt.subplots(1, 3, figsize=(19, 4.3))
        m0 = axes[0].pcolormesh(lon, lat, ng, cmap=fcmap, vmin=vmin, vmax=vmax,
                                shading="auto")
        axes[0].set_title(f"NeuralGCM day-10 climatology\n(mean of {out.attrs['n_inits']} forecasts)")
        fig.colorbar(m0, ax=axes[0], shrink=0.8, label=f"{label} [{units}]")
        m1 = axes[1].pcolormesh(lon, lat, rf, cmap=fcmap, vmin=vmin, vmax=vmax,
                                shading="auto")
        axes[1].set_title(f"{C.REF_LABEL} reference mean\n(reference)")
        fig.colorbar(m1, ax=axes[1], shrink=0.8, label=f"{label} [{units}]")
        m2 = axes[2].pcolormesh(lon, lat, dr, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                                shading="auto")
        gm = float(dr.weighted(np.cos(np.deg2rad(lat))).mean(["longitude", "latitude"]))
        axes[2].set_title(f"10-day drift = NeuralGCM − {C.REF_LABEL}\n({area} mean {gm:+.4g} {units})")
        fig.colorbar(m2, ax=axes[2], shrink=0.8, label=f"drift [{units}]")
        for ax in axes:
            C.draw_coastlines(ax)
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
            ax.set_xlim(w, e); ax.set_ylim(s, n_); ax.grid(alpha=0.2)
        fig.suptitle(f"{C.REF_LABEL} — mean 10-day {label}@{lev} hPa drift, {area} "
                     f"(Rackow et al. 2024, Fig. 3 style)", y=1.04, fontsize=13)
        fig.tight_layout()
        out_png = figdir / f"{C.RUN}_driftmap_L{lev:04d}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_variable(var, levels, regions, periods):
    meta = C.VARIABLES[var]
    short, units, label, fcmap = (meta["short"], meta["units"],
                                  meta["label"], meta["cmap"])
    print(f"=== drift maps: {var} ({short}) ===")
    truth = C.truth_at_levels(var, levels)
    for period in periods:
        out = build_fields(var, short, levels, truth, period)
        if out is None:
            continue
        for reg in regions:
            plot_period_region(out, var, short, units, label, fcmap, levels, period, reg)
        out.close()
        print(f"  saved {C.period_dir_name(period)} x {len(regions)} region(s)")


def main():
    levels = C.requested_levels()
    regions = C.selected_regions()
    periods = C.selected_periods()
    for var in C.selected_variables():
        plot_variable(var, levels, regions, periods)
    print(f"done -> {C.FIGROOT}/<period>/<region>/<variable>/drift_maps/")


if __name__ == "__main__":
    main()
