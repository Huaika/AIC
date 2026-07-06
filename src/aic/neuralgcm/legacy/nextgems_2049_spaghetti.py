#!/usr/bin/env python
"""Rackow et al. (2024) Fig. 1-style spaghetti plots for NextGEMS-2049.

Continuous NextGEMS-2049 daily-mean global temperature (thick black) with one
thin 10-day/6 h rollout line per start date (collapsed to per-day means), drawn
for every requested pressure level (see ng2049_common.requested_levels()).

2049 is a future-climate run with no ERA5 truth, so the reference is NextGEMS-2049
itself -- the dataset that initialised and forced the rollouts.

One full-year figure + one January zoom per level, saved under figures/spaghetti/.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import ng2049_common as C

YEAR = C.YEAR
FIGDIR = C.figure_dir("spaghetti")


def build_rollout_gmean(levels) -> pd.DataFrame:
    csv = C.OUTDIR / f"nextgems_{YEAR}_rollout_gmean_{C.level_tag()}.csv"
    if csv.exists():
        print(f"[rollout] cached {csv}")
        return pd.read_csv(csv, parse_dates=["init_date"])
    files = sorted(C.PRED_DIR.glob(f"pred_{YEAR}_*.nc"))
    print(f"[rollout] global-mean T at {len(levels)} levels from {len(files)} files")
    rows = []
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        init = pd.to_datetime(ds.attrs.get("init_date",
                                           f.stem.replace(f"pred_{YEAR}_", "")))
        gm = C.lat_weighted_mean(ds["temperature"].sel(level=levels)).compute()
        lead_h = ds["lead_hours"].values.astype(int)
        for lev in levels:
            rows.append(pd.DataFrame({
                "init_date": init, "lead_hours": lead_h, "level": lev,
                "t_pred_gmean_k": gm.sel(level=lev).values}))
        ds.close()
        if i % 50 == 0 or i == len(files) - 1:
            print(f"  {i + 1}/{len(files)}")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(csv, index=False)
    print(f"[rollout] wrote {csv}")
    return df


def spaghetti(ax, roll_lev, ref_daily_lev, level, color, every=1):
    r = roll_lev.copy()
    r["valid_time"] = r["init_date"] + pd.to_timedelta(r["lead_hours"], unit="h")
    r["lead_day_idx"] = (r["lead_hours"] // 24).astype(int)

    ax.plot(ref_daily_lev["date"], ref_daily_lev["t_ref_gmean_k"],
            color="black", lw=2.2, zorder=1, label=f"NextGEMS-{YEAR} (daily mean)")

    lw, alpha = (0.5, 0.5) if every == 1 else (0.9, 0.8)
    for d in sorted(r["init_date"].unique())[::every]:
        g = r[r["init_date"] == d]
        daily = (g.groupby("lead_day_idx")
                   .agg(vt=("valid_time", "mean"), t=("t_pred_gmean_k", "mean"))
                   .reset_index())
        ax.plot(daily["vt"], daily["t"], color=color, lw=lw, alpha=alpha, zorder=2)
    cadence = "every day" if every == 1 else f"every {every}th day"
    ax.plot([], [], color=color, lw=1.2, alpha=0.9,
            label=f"NeuralGCM 10-day rollout, daily mean ({cadence})")
    ax.set_title(f"Global-mean temperature at {level} hPa — NextGEMS-{YEAR}")
    ax.set_ylabel(f"T@{level}hPa global mean [K]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.margins(x=0.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)


def main():
    levels = C.requested_levels()
    roll = build_rollout_gmean(levels)

    # reference daily-mean global T per level, from the model-grid truth cache
    truth = C.truth_at_levels(levels)
    ref_gm = C.lat_weighted_mean(truth)                    # (time, level)
    ref = ref_gm.to_dataframe(name="t_ref_gmean_k").reset_index()
    ref["date"] = pd.to_datetime(ref["time"]).dt.floor("D")
    ref = ref.groupby(["date", "level"], as_index=False)["t_ref_gmean_k"].mean()

    cmap = plt.get_cmap("turbo")
    for k, lev in enumerate(levels):
        color = cmap(k / max(1, len(levels) - 1))
        roll_lev = roll[roll["level"] == lev]
        ref_lev = ref[ref["level"] == lev].sort_values("date")

        fig, ax = plt.subplots(figsize=(13, 5))
        spaghetti(ax, roll_lev, ref_lev, lev, color, every=1)
        ax.set_xlabel("Valid time")
        fig.tight_layout()
        out = FIGDIR / f"nextgems{YEAR}_spaghetti_T_L{lev:04d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(13, 5))
        spaghetti(ax, roll_lev, ref_lev, lev, color, every=1)
        ax.set_xlim(dt.datetime(YEAR, 1, 1), dt.datetime(YEAR, 1, 31))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set_title(f"Global-mean T at {lev} hPa — NextGEMS-{YEAR} (zoom: Jan)")
        ax.set_xlabel("Valid time")
        fig.tight_layout()
        out = FIGDIR / f"nextgems{YEAR}_spaghetti_T_L{lev:04d}_zoom.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"saved spaghetti L{lev}")
    print(f"done -> {FIGDIR}/")


if __name__ == "__main__":
    main()
