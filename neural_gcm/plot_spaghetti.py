#!/usr/bin/env python
"""Spaghetti plots (Rackow et al. 2024 Fig. 1 style) -- RUN- and VARIABLE-agnostic.

Continuous reference daily-mean global field (thick black) + one thin 10-day/6 h
rollout line per init-day (collapsed to per-day means), per requested pressure
level. One full-year figure per (variable, level). NO January zoom.

Run selected via EVAL_RUN; variable set via EVAL_VARS (see eval_common.py).
"""
from __future__ import annotations

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import eval_common as C

YEAR = C.YEAR
FIGDIR = C.figure_dir("spaghetti")


def build_rollout_gmean(var, short, levels) -> pd.DataFrame:
    csv = C.OUTDIR / f"{C.RUN}_rollout_gmean_{short}_{C.level_tag()}.csv"
    if csv.exists():
        print(f"[rollout] cached {csv}")
        return pd.read_csv(csv, parse_dates=["init_date"])
    files = sorted(C.PRED_DIR.glob(f"pred_{YEAR}_*.nc"))
    print(f"[rollout] global-mean {var} at {len(levels)} levels from {len(files)} files")
    rows = []
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        init = pd.to_datetime(ds.attrs.get("init_date",
                                           f.stem.replace(f"pred_{YEAR}_", "")))
        gm = C.lat_weighted_mean(ds[var].sel(level=levels)).compute()
        lead_h = ds["lead_hours"].values.astype(int)
        for lev in levels:
            rows.append(pd.DataFrame({
                "init_date": init, "lead_hours": lead_h, "level": lev,
                "pred_gmean": gm.sel(level=lev).values}))
        ds.close()
        if i % 50 == 0 or i == len(files) - 1:
            print(f"  {i + 1}/{len(files)}")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(csv, index=False)
    print(f"[rollout] wrote {csv}")
    return df


def spaghetti(ax, roll_lev, ref_daily_lev, level, color, label, units, every=1):
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
    ax.set_title(f"Global-mean {label} at {level} hPa — {C.REF_LABEL}")
    ax.set_ylabel(f"{label} @{level}hPa global mean [{units}]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.margins(x=0.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)


def plot_variable(var, levels):
    meta = C.VARIABLES[var]
    short, units, label = meta["short"], meta["units"], meta["label"]
    print(f"=== spaghetti: {var} ({short}) ===")
    roll = build_rollout_gmean(var, short, levels)

    truth = C.truth_at_levels(var, levels)
    ref_gm = C.lat_weighted_mean(truth)
    ref = ref_gm.to_dataframe(name="ref_gmean").reset_index()
    ref["date"] = pd.to_datetime(ref["time"]).dt.floor("D")
    ref = ref.groupby(["date", "level"], as_index=False)["ref_gmean"].mean()

    cmap = plt.get_cmap("turbo")
    for k, lev in enumerate(levels):
        color = cmap(k / max(1, len(levels) - 1))
        roll_lev = roll[roll["level"] == lev]
        ref_lev = ref[ref["level"] == lev].sort_values("date")

        fig, ax = plt.subplots(figsize=(13, 5))
        spaghetti(ax, roll_lev, ref_lev, lev, color, label, units, every=1)
        ax.set_xlabel("Valid time")
        fig.tight_layout()
        out = FIGDIR / f"{C.RUN}_spaghetti_{short}_L{lev:04d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"saved spaghetti {short} L{lev}")


def main():
    levels = C.requested_levels()
    for var in C.selected_variables():
        plot_variable(var, levels)
    print(f"done -> {FIGDIR}/")


if __name__ == "__main__":
    main()
