#!/usr/bin/env python
"""Drift statistics (year-mean RMSE + bias vs lead) -- RUN/VARIABLE/REGION-agnostic.

Pipeline
--------
For every rollout file (one per init-day ``d``) and every lead step ``s``:

1. ``diff = pred(d, s) - ref(valid_time)`` on the model grid,
2. per region: ``mse(d, s) = <diff^2>`` and ``bias(d, s) = <diff>``
   (area-weighted mean on the region box),
3. cache these per-(init, lead, level, region) rows in a tidy CSV.

Then, for each (period, region, level), aggregate across all inits into
``rmse = sqrt(mean_d mse)`` and ``mean_d bias`` vs lead-day, and draw one
twin-axis figure (blue = RMSE, red = bias).

For ERA5 runs the reference is real truth, so these are genuine forecast-skill
curves; for NextGEMS-2049 the reference is NextGEMS itself (pure drift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ngcm import eval_common as C
from ngcm.plots import _helpers as H

YEAR = C.YEAR


def build_drift(var, short, levels, truth, regions) -> pd.DataFrame:
    """Per-(init, lead, level, region) MSE and bias frame, cached to CSV."""
    csv = C.OUTDIR / f"{C.RUN}_drift_per_init_{short}_{C.level_tag()}.csv"
    cached = H.load_region_cache(csv, regions, tag="drift")
    if cached is not None:
        return cached

    files = H.prediction_files()
    print(f"[drift] scoring {len(files)} rollouts vs {C.REF_LABEL} "
          f"({var}) at {len(levels)} levels, {len(regions)} region(s)")

    rows = []
    for _, f, ds in H.iter_predictions(files, tag="drift"):
        init = H.pred_init_date(ds, f)
        pred = ds[var].sel(level=levels)
        tru = truth.sel(time=ds["valid_time"].values, method="nearest")
        tru = tru.assign_coords(time=pred["time"].values)
        diff = (pred - tru).compute()
        lead_h = ds["lead_hours"].values.astype(int)
        for reg in regions:
            dr = C.select_region(diff, reg)
            mse = C.lat_weighted_mean(dr ** 2)
            bias = C.lat_weighted_mean(dr)
            for lev in levels:
                rows.append(pd.DataFrame({
                    "init_date": init, "lead_hours": lead_h, "level": lev,
                    "region": reg,
                    "mse": np.asarray(mse.sel(level=lev).values, float),
                    "bias": np.asarray(bias.sel(level=lev).values, float)}))
        ds.close()

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(csv, index=False)
    print(f"[drift] wrote {csv}")
    return df


def aggregate(df, short, period) -> pd.DataFrame:
    """Collapse per-init rows to year-mean ``rmse`` / ``bias`` vs lead."""
    agg = (df.groupby(["region", "level", "lead_hours"], as_index=False)
           .agg(mse=("mse", "mean"), bias=("bias", "mean"),
                n_init=("init_date", "nunique")))
    agg["rmse"] = np.sqrt(agg["mse"])
    agg["lead_day"] = agg["lead_hours"] / 24.0
    suffix = "" if period == 0 else f"_{period:02d}"
    agg.to_csv(C.OUTDIR / f"{C.RUN}_drift_yearmean_{short}{suffix}_{C.level_tag()}.csv",
               index=False)
    return agg


def _plot_rmse_bias(a, level, label, units, region, period, var, figdir):
    """One twin-axis RMSE (blue) + bias (red) figure for a single level."""
    fig, ax_rmse = plt.subplots(figsize=(6.5, 4.4))
    ax_bias = ax_rmse.twinx()

    ax_rmse.plot(a["lead_day"], a["rmse"], color="#1f77b4", label="RMSE")
    ax_bias.plot(a["lead_day"], a["bias"], color="#d62728", label="bias")
    ax_bias.axhline(0.0, color="#d62728", lw=0.8, ls=":", alpha=0.6)

    ax_rmse.set_title(f"{C.REF_LABEL} — {level} hPa {label} ({H.area_name(region)}, "
                      f"mean of {int(a['n_init'].iloc[0])} daily inits)")
    ax_rmse.set_xlabel("lead time (days)")
    ax_rmse.set_ylabel(f"RMSE [{units}]", color="#1f77b4")
    ax_bias.set_ylabel(f"mean bias [{units}]", color="#d62728")
    ax_rmse.tick_params(axis="y", labelcolor="#1f77b4")
    ax_bias.tick_params(axis="y", labelcolor="#d62728")
    ax_rmse.grid(True, alpha=0.3)
    fig.tight_layout()
    H.save_and_close(fig, figdir, region, var, level, period, "drift_stats")


def plot_variable(var, levels, regions, periods):
    meta = C.VARIABLES[var]
    short, units, label = meta["short"], meta["units"], meta["label"]
    print(f"=== drift stats: {var} ({short}) ===")

    truth = C.truth_at_levels(var, levels)
    df = build_drift(var, short, levels, truth, regions)
    df["_m"] = df["init_date"].dt.month

    for period in periods:
        sub = df if period == 0 else df[df["_m"] == period]
        if sub.empty:
            print(f"  [skip] no init-days in {C.period_dir_name(period)}")
            continue
        agg = aggregate(sub, short, period)
        for reg in regions:
            figdir = C.figure_dir(period, reg, var, "drift_stats")
            ar = agg[agg["region"] == reg]
            for lev in C.render_levels(levels):
                a = ar[ar["level"] == lev].sort_values("lead_hours")
                _plot_rmse_bias(a, lev, label, units, reg, period, var, figdir)
        print(f"  saved {C.period_dir_name(period)} x {len(regions)} region(s)")


def main():
    H.run_for_variables(plot_variable, "drift_stats")


if __name__ == "__main__":
    main()
