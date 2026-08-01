#!/usr/bin/env python
"""Drift statistics (year-mean RMSE + bias vs lead) -- multi-source aware.

For each init-day d and lead step s:  diff = pred(d,s) - ref(valid_time) on the
model grid; mse(s)=mean_d<diff^2>, bias(s)=mean_d<diff>, RMSE=sqrt(mse), all
area-weighted over the region. One twin-axis figure per (region, variable, level).

Single source (EVAL_RUN, or one model in EVAL_SOURCES): the classic blue-RMSE /
red-bias twin-axis plot (byte-identical name/paths to before). Multiple sources
(EVAL_SOURCES=neuralgcm,graphcast + EVAL_YEAR): one RMSE curve (solid) + one bias
curve (dashed) PER source, coloured per source, on shared twin axes.

For ERA5 runs the reference is real truth (genuine forecast-skill curves); for
NextGEMS-2049 the reference is NextGEMS itself (drift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aic.controller.eval import eval_common as C
from aic.controller.eval import sources as S
from aic.view import naming as fig_naming


def aggregate(df) -> pd.DataFrame:
    agg = (df.groupby(["region", "level", "lead_hours"], as_index=False)
             .agg(mse=("mse", "mean"), bias=("bias", "mean"),
                  n_init=("init_date", "nunique")))
    agg["rmse"] = np.sqrt(agg["mse"])
    agg["lead_day"] = agg["lead_hours"] / 24.0
    return agg


def plot_variable(sources, var, levels, regions, periods):
    meta = C.VARIABLES[var]
    short, units, label = meta["short"], meta["units"], meta["label"]
    print(f"=== drift stats: {var} ({short}) ===")
    single = not S.is_multi(sources)
    # per-source per-init drift, tagged with month for period slicing
    per = {}
    for s in sources:
        d = s.drift_per_init(var, short, levels, regions)
        d["_m"] = d["init_date"].dt.month
        per[s.run] = d

    for period in periods:
        aggs = {}
        for s in sources:
            d = per[s.run]
            sub = d if period == 0 else d[d["_m"] == period]
            if not sub.empty:
                aggs[s.run] = aggregate(sub)
        if not aggs:
            print(f"  [skip] no init-days in {C.period_dir_name(period)}")
            continue
        for reg in regions:
            figdir = S.figure_dir(sources, period, reg, var, "drift_stats")
            area = "global" if reg == "world" else reg
            for lev in C.render_levels(levels):
                fig, ax_rmse = plt.subplots(figsize=(6.5, 4.4))
                ax_bias = ax_rmse.twinx()
                n_ref = None
                for s in sources:
                    if s.run not in aggs:
                        continue
                    ar = aggs[s.run]
                    a = ar[(ar["region"] == reg) & (ar["level"] == lev)] \
                        .sort_values("lead_hours")
                    if a.empty:
                        continue
                    n_ref = int(a["n_init"].iloc[0])
                    if single:
                        rc, bc = "#1f77b4", "#d62728"
                        rlab, blab = "RMSE", "bias"
                    else:
                        rc = bc = s.color
                        rlab, blab = f"{s.pretty} RMSE", f"{s.pretty} bias"
                    ax_rmse.plot(a["lead_day"], a["rmse"], color=rc, lw=1.8, label=rlab)
                    ax_bias.plot(a["lead_day"], a["bias"], color=bc, lw=1.4,
                                 ls="-" if single else "--", label=blab)
                ax_bias.axhline(0.0, color="0.4", lw=0.8, ls=":", alpha=0.6)
                title_model = sources[0].ref_label if single else \
                    " vs ".join(dict.fromkeys(s.pretty for s in sources))
                ax_rmse.set_title(f"{title_model} — {lev} hPa {label} ({area}, "
                                  f"mean of {n_ref} daily inits)")
                ax_rmse.set_xlabel("lead time (days)")
                if single:
                    ax_rmse.set_ylabel(f"RMSE [{units}]", color="#1f77b4")
                    ax_bias.set_ylabel(f"mean bias [{units}]", color="#d62728")
                    ax_rmse.tick_params(axis="y", labelcolor="#1f77b4")
                    ax_bias.tick_params(axis="y", labelcolor="#d62728")
                else:
                    ax_rmse.set_ylabel(f"RMSE [{units}] (solid)")
                    ax_bias.set_ylabel(f"mean bias [{units}] (dashed)")
                    h1, l1 = ax_rmse.get_legend_handles_labels()
                    h2, l2 = ax_bias.get_legend_handles_labels()
                    ax_rmse.legend(h1 + h2, l1 + l2, loc="upper left",
                                   fontsize=8, framealpha=0.9)
                ax_rmse.grid(True, alpha=0.3)
                fig.tight_layout()
                out = figdir / fig_naming.figure_name(
                    S.model_token(sources), sources[0].dataset, reg, var, lev,
                    sources[0].year, fig_naming.months_token(period),
                    "drift_stats", ext="pdf")
                fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {C.period_dir_name(period)} x {len(regions)} region(s)")


def main():
    sources = S.resolve_sources()
    levels = S.requested_levels(sources)
    regions = C.selected_regions()
    periods = C.selected_periods()
    for var in C.selected_variables():
        plot_variable(sources, var, levels, regions, periods)
    print(f"done -> {C.FIG_ROOT}/{S.run_label(sources)}/<period>/<region>/<variable>/drift_stats/")


if __name__ == "__main__":
    main()
