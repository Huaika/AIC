#!/usr/bin/env python
"""Drift statistics (year-mean RMSE + bias vs lead time) -- RUN/VARIABLE-agnostic.

For each init-day d and lead step s:  diff = pred(d,s) - ref(valid_time) on the
model grid; mse(s)=mean_d<diff^2>, bias(s)=mean_d<diff>, RMSE=sqrt(mse). One
twin-axis figure per (variable, level). Run via EVAL_RUN, variables via EVAL_VARS.

For ERA5 runs the reference is real truth, so these are genuine forecast-skill
curves; for NextGEMS-2049 the reference is NextGEMS itself (drift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import eval_common as C

YEAR = C.YEAR
FIGDIR = C.figure_dir("drift_stats")


def build_drift(var, short, levels, truth) -> pd.DataFrame:
    csv = C.OUTDIR / f"{C.RUN}_drift_per_init_{short}_{C.level_tag()}.csv"
    if csv.exists():
        print(f"[drift] cached {csv}")
        return pd.read_csv(csv, parse_dates=["init_date"])
    files = sorted(C.PRED_DIR.glob(f"pred_{YEAR}_*.nc"))
    print(f"[drift] scoring {len(files)} rollouts vs {C.REF_LABEL} "
          f"({var}) at {len(levels)} levels")
    rows = []
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        init = pd.to_datetime(ds.attrs.get("init_date",
                                           f.stem.replace(f"pred_{YEAR}_", "")))
        pred = ds[var].sel(level=levels)
        tru = truth.sel(time=ds["valid_time"].values, method="nearest")
        tru = tru.assign_coords(time=pred["time"].values)
        diff = pred - tru
        mse = C.lat_weighted_mean(diff ** 2)
        bias = C.lat_weighted_mean(diff)
        lead_h = ds["lead_hours"].values.astype(int)
        for lev in levels:
            rows.append(pd.DataFrame({
                "init_date": init, "lead_hours": lead_h, "level": lev,
                "mse": np.asarray(mse.sel(level=lev).values, float),
                "bias": np.asarray(bias.sel(level=lev).values, float)}))
        ds.close()
        if i % 50 == 0 or i == len(files) - 1:
            print(f"  {i + 1}/{len(files)}")
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(csv, index=False)
    print(f"[drift] wrote {csv}")
    return df


def aggregate(df, short) -> pd.DataFrame:
    agg = (df.groupby(["level", "lead_hours"], as_index=False)
             .agg(mse=("mse", "mean"), bias=("bias", "mean"),
                  n_init=("init_date", "nunique")))
    agg["rmse"] = np.sqrt(agg["mse"])
    agg["lead_day"] = agg["lead_hours"] / 24.0
    agg.to_csv(C.OUTDIR / f"{C.RUN}_drift_yearmean_{short}_{C.level_tag()}.csv",
               index=False)
    return agg


def plot_variable(var, levels):
    meta = C.VARIABLES[var]
    short, units, label = meta["short"], meta["units"], meta["label"]
    print(f"=== drift stats: {var} ({short}) ===")
    truth = C.truth_at_levels(var, levels)
    agg = aggregate(build_drift(var, short, levels, truth), short)

    for lev in levels:
        a = agg[agg["level"] == lev].sort_values("lead_hours")
        fig, ax_rmse = plt.subplots(figsize=(6.5, 4.4))
        ax_bias = ax_rmse.twinx()
        ax_rmse.plot(a["lead_day"], a["rmse"], color="#1f77b4", label="RMSE")
        ax_bias.plot(a["lead_day"], a["bias"], color="#d62728", label="bias")
        ax_bias.axhline(0.0, color="#d62728", lw=0.8, ls=":", alpha=0.6)
        ax_rmse.set_title(f"{C.REF_LABEL} — {lev} hPa {label} "
                          f"(mean of {int(a['n_init'].iloc[0])} daily inits)")
        ax_rmse.set_xlabel("lead time (days)")
        ax_rmse.set_ylabel(f"RMSE [{units}]", color="#1f77b4")
        ax_bias.set_ylabel(f"mean bias [{units}]", color="#d62728")
        ax_rmse.tick_params(axis="y", labelcolor="#1f77b4")
        ax_bias.tick_params(axis="y", labelcolor="#d62728")
        ax_rmse.grid(True, alpha=0.3)
        fig.tight_layout()
        out = FIGDIR / f"{C.RUN}_drift_rmse_bias_{short}_L{lev:04d}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"{short} L{lev}: day-10 RMSE={a['rmse'].iloc[-1]:.4g} {units} "
              f"bias={a['bias'].iloc[-1]:+.4g} {units}")


def main():
    levels = C.requested_levels()
    for var in C.selected_variables():
        plot_variable(var, levels)
    print(f"done -> {FIGDIR}/")


if __name__ == "__main__":
    main()
