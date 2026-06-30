#!/usr/bin/env python
"""Spaghetti plots (Rackow et al. 2024 Fig. 1 style) -- RUN/VARIABLE/REGION-agnostic.

Continuous reference daily-mean area-mean field (thick black) + one thin 10-day/6 h
rollout line per init-day (collapsed to per-day means), per requested pressure
level. One full-year figure per (region, variable, level).

Run via EVAL_RUN; variables via EVAL_VARS; regions via EVAL_REGIONS (default
world). All requested regions are computed in ONE pass over the prediction files
(region is a column in the cached CSV). Output:
figures/<run>/<region>/<variable>/spaghetti/.
"""
from __future__ import annotations

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import eval_common as C

YEAR = C.YEAR


def build_rollout_gmean(var, short, levels, regions) -> pd.DataFrame:
    csv = C.OUTDIR / f"{C.RUN}_rollout_gmean_{short}_{C.level_tag()}.csv"
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["init_date"])
        if "region" in df.columns and set(regions) <= set(df["region"].unique()):
            print(f"[rollout] cached {csv}")
            return df
        print(f"[rollout] cache {csv} missing regions -> recompute")
    files = sorted(C.PRED_DIR.glob(f"pred_{YEAR}_*.nc"))
    print(f"[rollout] area-mean {var} at {len(levels)} levels, {len(regions)} "
          f"region(s), from {len(files)} files")
    rows = []
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        init = pd.to_datetime(ds.attrs.get("init_date",
                                           f.stem.replace(f"pred_{YEAR}_", "")))
        da = ds[var].sel(level=levels).compute()
        lead_h = ds["lead_hours"].values.astype(int)
        for reg in regions:
            gm = C.lat_weighted_mean(C.select_region(da, reg))
            for lev in levels:
                rows.append(pd.DataFrame({
                    "init_date": init, "lead_hours": lead_h, "level": lev,
                    "region": reg, "pred_gmean": gm.sel(level=lev).values}))
        ds.close()
        if i % 50 == 0 or i == len(files) - 1:
            print(f"  {i + 1}/{len(files)}")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(csv, index=False)
    print(f"[rollout] wrote {csv}")
    return df


def build_ref(var, levels, regions) -> pd.DataFrame:
    """Reference daily area-mean per region, as a tidy frame."""
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
    area = "global" if region == "world" else region
    ax.set_title(f"{area.capitalize()}-mean {label} at {level} hPa — {C.REF_LABEL}")
    ax.set_ylabel(f"{label} @{level}hPa {area} mean [{units}]")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.margins(x=0.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)


def plot_variable(var, levels, regions):
    meta = C.VARIABLES[var]
    short, units, label = meta["short"], meta["units"], meta["label"]
    print(f"=== spaghetti: {var} ({short}) ===")
    roll = build_rollout_gmean(var, short, levels, regions)
    ref = build_ref(var, levels, regions)

    cmap = plt.get_cmap("turbo")
    for reg in regions:
        figdir = C.figure_dir(reg, var, "spaghetti")
        roll_r = roll[roll["region"] == reg]
        ref_r = ref[ref["region"] == reg]
        for k, lev in enumerate(levels):
            color = cmap(k / max(1, len(levels) - 1))
            roll_lev = roll_r[roll_r["level"] == lev]
            ref_lev = ref_r[ref_r["level"] == lev].sort_values("date")

            fig, ax = plt.subplots(figsize=(13, 5))
            spaghetti(ax, roll_lev, ref_lev, lev, color, label, units, reg, every=1)
            ax.set_xlabel("Valid time")
            fig.tight_layout()
            out = figdir / f"{C.RUN}_spaghetti_L{lev:04d}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {reg}/{var} ({len(levels)} levels)")


def main():
    levels = C.requested_levels()
    regions = C.selected_regions()
    for var in C.selected_variables():
        plot_variable(var, levels, regions)
    print(f"done -> {C.FIGROOT}/<region>/<variable>/spaghetti/")


if __name__ == "__main__":
    main()
