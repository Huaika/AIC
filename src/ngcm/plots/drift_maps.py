#!/usr/bin/env python
"""10-day drift maps (Rackow et al. 2024 Fig. 3 style) -- RUN/VARIABLE/REGION-agnostic.

For each (region, variable, level) draw a 3-panel figure:

1. ``ngcm_day10_clim`` -- mean, over all inits in the period, of that rollout's
   final-day (leads 216..240 h) field,
2. ``ref_annual_clim`` -- reference annual (or per-month) mean field,
3. ``drift = ngcm_day10_clim - ref_annual_clim`` on the model grid.

The clim / drift fields are computed GLOBALLY and cached once per
(run, variable, period); each region is just that global field cropped to the
region box (no recompute). Output:
``figures/<run>/<region>/<variable>/drift_maps/``.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ngcm import eval_common as C
from ngcm.plots import _helpers as H

YEAR = C.YEAR
FINAL_DAY_LEAD_MIN = 216  # h; leads >= this are the "day-10" fields


def build_fields(var, short, levels, truth, period) -> xr.Dataset | None:
    """Day-10 clim / reference clim / drift fields for a period, cached to NetCDF.

    ``period == 0`` averages over the whole year; ``period == m`` averages only
    inits (and truth times) in month ``m``. Returns ``None`` if the period has
    no init-days.
    """
    suffix = "" if period == 0 else f"_{period:02d}"
    nc = C.OUTDIR / f"{C.RUN}_drift_maps_{short}{suffix}_{C.level_tag()}.nc"
    if nc.exists():
        print(f"[maps] cached {nc}")
        return xr.open_dataset(nc)

    files = H.prediction_files()
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
    for _, _, ds in H.iter_predictions(files, tag="maps"):
        t = ds[var].sel(level=levels)
        end = t.where(ds["lead_hours"] >= FINAL_DAY_LEAD_MIN, drop=True).mean("time")
        end = end.transpose("level", "latitude", "longitude")
        acc = end if acc is None else acc + end
        n += 1
        ds.close()

    ngcm = acc / n
    out = xr.Dataset({"ngcm_day10_clim": ngcm, "ref_annual_clim": ref_clim,
                      "drift": ngcm - ref_clim})
    out.attrs.update(n_inits=n, final_day_lead_min_h=FINAL_DAY_LEAD_MIN,
                     variable=var, period=C.period_dir_name(period),
                     drift_def=f"mean(end-of-10day-forecast) - {C.REF_LABEL} mean")
    out.to_netcdf(nc)
    print(f"[maps] wrote {nc}")
    return out


def plot_period_region(out, var, units, label, fcmap, levels, period, reg):
    """Draw the 3-panel drift figure for one (period, region), for every level."""
    figdir = C.figure_dir(period, reg, var, "drift_maps")
    extent = C.region_extent(reg)
    area = H.area_name(reg)

    for lev in levels:
        ng = C.select_region(out["ngcm_day10_clim"].sel(level=lev), reg)
        rf = C.select_region(out["ref_annual_clim"].sel(level=lev), reg)
        dr = C.select_region(out["drift"].sel(level=lev), reg)
        lon, lat = ng.longitude, ng.latitude

        # Area-weighted mean drift for the diff-panel title.
        gm = float(dr.weighted(np.cos(np.deg2rad(lat))).mean(["longitude", "latitude"]))
        titles = (
            f"NeuralGCM day-10 climatology\n(mean of {out.attrs['n_inits']} forecasts)",
            f"{C.REF_LABEL} reference mean\n(reference)",
            f"10-day drift = NeuralGCM − {C.REF_LABEL}\n({area} mean {gm:+.4g} {units})",
        )

        fig, axes = plt.subplots(1, 3, figsize=(19, 4.3))
        H.plot_three_panel_maps(
            axes, fig, lon, lat, ng, rf, dr,
            field_cmap=fcmap, units=units, field_label=label,
            diff_label="drift", titles=titles, extent=extent,
        )
        fig.suptitle(f"{C.REF_LABEL} — mean 10-day {label}@{lev} hPa drift, {area} "
                     f"(Rackow et al. 2024, Fig. 3 style)", y=1.04, fontsize=13)
        fig.tight_layout()
        H.save_and_close(fig, figdir, reg, var, lev, period, "drift_maps", ext="png")


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
            plot_period_region(out, var, units, label, fcmap, levels, period, reg)
        out.close()
        print(f"  saved {C.period_dir_name(period)} x {len(regions)} region(s)")


def main():
    H.run_for_variables(plot_variable, "drift_maps")


if __name__ == "__main__":
    main()
