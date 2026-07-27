#!/usr/bin/env python
"""Spaghetti plots (Rackow et al. 2024 Fig. 1 style) -- RUN/VARIABLE/REGION-agnostic.

For every requested pressure level, draw a single year-long figure per (region,
variable, level):

* a thick black line = reference daily-mean, area-mean field (the "truth");
* one thin colored line per init-day = that day's 10-day / 6 h rollout,
  collapsed to daily means and plotted against valid time.

All requested regions are computed in ONE pass over the prediction files;
``region`` is a column in the cached CSV so re-runs skip the reload. Output:
``figures/<run>/<region>/<variable>/spaghetti/``.
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ngcm import eval_common as C
from ngcm.plots import _helpers as H

YEAR = C.YEAR


def build_rollout_gmean(var, short, levels, regions) -> pd.DataFrame:
    """Per-(init, lead, level, region) area-mean of the rollout, cached to CSV."""
    csv = C.OUTDIR / f"{C.RUN}_rollout_gmean_{short}_{C.level_tag()}.csv"
    cached = H.load_region_cache(csv, regions, tag="rollout")
    if cached is not None:
        return cached

    files = H.prediction_files()
    print(f"[rollout] area-mean {var} at {len(levels)} levels, {len(regions)} "
          f"region(s), from {len(files)} files")

    rows = []
    for _, f, ds in H.iter_predictions(files, tag="rollout"):
        init = H.pred_init_date(ds, f)
        da = ds[var].sel(level=levels).compute()
        lead_h = ds["lead_hours"].values.astype(int)
        for reg in regions:
            gm = C.lat_weighted_mean(C.select_region(da, reg))
            for lev in levels:
                rows.append(pd.DataFrame({
                    "init_date": init, "lead_hours": lead_h, "level": lev,
                    "region": reg, "pred_gmean": gm.sel(level=lev).values}))
        ds.close()

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(csv, index=False)
    print(f"[rollout] wrote {csv}")
    return df


def build_ref(var, levels, regions) -> pd.DataFrame:
    """Reference daily area-mean per region as a tidy frame."""
    truth = C.truth_at_levels(var, levels)
    frames = []
    for reg in regions:
        ref_gm = C.lat_weighted_mean(C.select_region(truth, reg))
        d = ref_gm.to_dataframe(name="ref_gmean").reset_index()
        d["region"] = reg
        d["date"] = pd.to_datetime(d["time"]).dt.floor("D")
        d = d.groupby(["region", "date", "level"], as_index=False)["ref_gmean"].mean()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def spaghetti(ax, roll_lev, ref_daily_lev, level, color, label, units, region, every=1):
    """Draw one (region, level) spaghetti panel onto ``ax``."""
    r = roll_lev.copy()
    r["valid_time"] = r["init_date"] + pd.to_timedelta(r["lead_hours"], unit="h")
    r["lead_day_idx"] = (r["lead_hours"] // 24).astype(int)

    ax.plot(ref_daily_lev["date"], ref_daily_lev["ref_gmean"],
            color="black", lw=2.2, zorder=1, label=f"{C.REF_LABEL} (daily mean)")

    lw, alpha = (0.5, 0.5) if every == 1 else (0.9, 0.8)
    for d in sorted(r["init_date"].unique())[::every]:
        g = r[r["init_date"] == d]
        daily = (g.groupby("lead_day_idx")
                 .agg(vt=("valid_time", "mean"), val=("pred_gmean", "mean"))
                 .reset_index())
        ax.plot(daily["vt"], daily["val"], color=color, lw=lw, alpha=alpha, zorder=2)

    cadence = "every day" if every == 1 else f"every {every}th day"
    ax.plot([], [], color=color, lw=1.2, alpha=0.9,
            label=f"NeuralGCM 10-day rollout, daily mean ({cadence})")

    area = H.area_name(region)
    ax.set_title(f"{area.capitalize()}-mean {label} at {level} hPa — {C.REF_LABEL}")
    ax.set_ylabel(f"{label} @{level}hPa {area} mean [{units}]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.margins(x=0.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)


def plot_variable(var, levels, regions, periods):
    meta = C.VARIABLES[var]
    units, label = meta["units"], meta["label"]
    print(f"=== spaghetti: {var} ({meta['short']}) ===")

    roll = build_rollout_gmean(var, meta["short"], levels, regions)
    ref = build_ref(var, levels, regions)
    roll["_m"] = roll["init_date"].dt.month
    ref["_m"] = pd.to_datetime(ref["date"]).dt.month

    cmap = plt.get_cmap("turbo")
    for period in periods:
        roll_p = roll if period == 0 else roll[roll["_m"] == period]
        ref_p = ref if period == 0 else ref[ref["_m"] == period]
        if roll_p.empty:
            print(f"  [skip] no init-days in {C.period_dir_name(period)}")
            continue
        for reg in regions:
            figdir = C.figure_dir(period, reg, var, "spaghetti")
            roll_r = roll_p[roll_p["region"] == reg]
            ref_r = ref_p[ref_p["region"] == reg]
            for k, lev in enumerate(C.render_levels(levels)):
                color = cmap(k / max(1, len(levels) - 1))
                roll_lev = roll_r[roll_r["level"] == lev]
                ref_lev = ref_r[ref_r["level"] == lev].sort_values("date")
                fig, ax = plt.subplots(figsize=(13, 5))
                spaghetti(ax, roll_lev, ref_lev, lev, color, label, units, reg, every=1)
                ax.set_xlabel("Valid time")
                fig.tight_layout()
                H.save_and_close(fig, figdir, reg, var, lev, period, "spaghetti")
        print(f"  saved {C.period_dir_name(period)} x {len(regions)} region(s)")


def main():
    H.run_for_variables(plot_variable, "spaghetti")


if __name__ == "__main__":
    main()
