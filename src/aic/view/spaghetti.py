#!/usr/bin/env python
"""Spaghetti plots (Rackow et al. 2024 Fig. 1 style) -- multi-source aware.

Continuous reference daily-mean area-mean field (thick black) + one thin 10-day/
6 h rollout line per init-day (collapsed to per-day means) FOR EACH forecast
source, per requested pressure level. One figure per (region, variable, level).

Sources are selected via EVAL_SOURCES (+ EVAL_YEAR), e.g.
    EVAL_SOURCES=neuralgcm,graphcast EVAL_YEAR=2023   # overlay both models
Unset EVAL_SOURCES -> single source from EVAL_RUN (historical behaviour, and the
figure name/paths are byte-identical to before). Variables via EVAL_VARS, regions
via EVAL_REGIONS (default world), periods via EVAL_MONTHS.
Output: figures/<run-or-compare>/<period>/<region>/<variable>/spaghetti/.
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from aic.controller.eval import eval_common as C
from aic.controller.eval import sources as S
from aic.view import naming as fig_naming


def build_ref(source, var, levels, regions) -> pd.DataFrame:
    """Reference daily area-mean per region, as a tidy frame (from one source's
    own-grid truth -- the overlay draws a single reference line)."""
    truth = source.truth_at_levels(var, levels)
    frames = []
    for reg in regions:
        ref_gm = C.lat_weighted_mean(C.select_region(truth, reg))
        d = ref_gm.to_dataframe(name="ref_gmean").reset_index()
        d["region"] = reg
        d["date"] = pd.to_datetime(d["time"]).dt.floor("D")
        d = d.groupby(["region", "date", "level"], as_index=False)["ref_gmean"].mean()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _draw_bundle(ax, roll_lev, color, label, every=1):
    """One forecast source's thin rollout lines (per init-day, daily-mean)."""
    r = roll_lev.copy()
    r["valid_time"] = r["init_date"] + pd.to_timedelta(r["lead_hours"], unit="h")
    r["lead_day_idx"] = (r["lead_hours"] // 24).astype(int)
    lw, alpha = (0.5, 0.5) if every == 1 else (0.9, 0.8)
    for d in sorted(r["init_date"].unique())[::every]:
        g = r[r["init_date"] == d]
        daily = (g.groupby("lead_day_idx")
                   .agg(vt=("valid_time", "mean"), val=("pred_gmean", "mean"))
                   .reset_index())
        ax.plot(daily["vt"], daily["val"], color=color, lw=lw, alpha=alpha, zorder=2)
    ax.plot([], [], color=color, lw=1.4, alpha=0.9, label=label)  # legend proxy


def plot_variable(sources, var, levels, regions, periods):
    meta = C.VARIABLES[var]
    short, units, label = meta["short"], meta["units"], meta["label"]
    print(f"=== spaghetti: {var} ({short}) ===")
    ref_src = sources[0]
    rolls = {s.run: s.rollout_gmean(var, short, levels, regions) for s in sources}
    ref = build_ref(ref_src, var, levels, regions)
    for df in rolls.values():
        df["_m"] = df["init_date"].dt.month
    ref["_m"] = pd.to_datetime(ref["date"]).dt.month

    for period in periods:
        ref_p = ref if period == 0 else ref[ref["_m"] == period]
        rolls_p = {run: (df if period == 0 else df[df["_m"] == period])
                   for run, df in rolls.items()}
        if all(df.empty for df in rolls_p.values()):
            print(f"  [skip] no init-days in {C.period_dir_name(period)}")
            continue
        for reg in regions:
            figdir = S.figure_dir(sources, period, reg, var, "spaghetti")
            ref_r = ref_p[ref_p["region"] == reg]
            area = "global" if reg == "world" else reg
            for lev in C.render_levels(levels):
                ref_lev = ref_r[ref_r["level"] == lev].sort_values("date")
                fig, ax = plt.subplots(figsize=(13, 5))
                ax.plot(ref_lev["date"], ref_lev["ref_gmean"], color="black",
                        lw=2.2, zorder=1, label=f"{ref_src.ref_label} (daily mean)")
                for s in sources:
                    roll_lev = rolls_p[s.run]
                    roll_lev = roll_lev[(roll_lev["region"] == reg)
                                        & (roll_lev["level"] == lev)]
                    if roll_lev.empty:
                        continue
                    _draw_bundle(ax, roll_lev, s.color,
                                 f"{s.pretty} 10-day rollout (daily mean)")
                ax.set_title(f"{area.capitalize()}-mean {label} at {lev} hPa "
                             f"— {ref_src.ref_label}")
                ax.set_ylabel(f"{label} @{lev}hPa {area} mean [{units}]")
                ax.set_xlabel("Valid time")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.margins(x=0.01)
                ax.grid(alpha=0.25)
                ax.legend(loc="upper left", framealpha=0.9)
                fig.tight_layout()
                out = figdir / fig_naming.figure_name(
                    S.model_token(sources), ref_src.dataset, reg, var, lev,
                    ref_src.year, fig_naming.months_token(period), "spaghetti",
                    ext="pdf")
                fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {C.period_dir_name(period)} x {len(regions)} region(s)")


def main():
    sources = S.resolve_sources()
    levels = S.requested_levels(sources)
    regions = C.selected_regions()
    periods = C.selected_periods()
    for var in C.selected_variables():
        plot_variable(sources, var, levels, regions, periods)
    print(f"done -> {C.FIG_ROOT}/{S.run_label(sources)}/<period>/<region>/<variable>/spaghetti/")


if __name__ == "__main__":
    main()
