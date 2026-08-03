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


def _doy_axis(dates):
    """Map a datetime series to a common leap reference year (2000) so different
    years overlay on one Jan-Dec axis."""
    doy = pd.to_datetime(dates).dayofyear
    return pd.Timestamp("2000-01-01") + pd.to_timedelta(doy - 1, unit="D")


def plot_variable_multiyear(sources, var, levels, regions):
    """Overlay several (model, year) rollout bundles + their references on ONE
    day-of-year axis (multi-year / out-of-distribution comparison): colour = model,
    line style = year. One reference line per (dataset, year) whose truth cache is
    available (missing ones are simply skipped)."""
    from matplotlib.lines import Line2D
    meta = C.VARIABLES[var]
    short, units, label = meta["short"], meta["units"], meta["label"]
    print(f"=== spaghetti (multi-year): {var} ({short}) ===")
    years = sorted({s.year for s in sources})
    ystyle = {y: S.year_linestyle(i) for i, y in enumerate(years)}
    rolls = {s.run: s.rollout_gmean(var, short, levels, regions) for s in sources}
    refs = {}                                        # (dataset, year) -> ref frame
    for s in sources:
        key = (s.dataset, s.year)
        if key in refs:
            continue
        try:
            refs[key] = build_ref(s, var, levels, regions)
        except SystemExit:
            print(f"  [ref] no truth cache for {s.run}; skipping its reference line")
            refs[key] = None

    for reg in regions:
        area = "global" if reg == "world" else reg
        for lev in C.render_levels(levels):
            fig, ax = plt.subplots(figsize=(13, 5.2))
            for (dset, yr), rf in refs.items():
                if rf is None:
                    continue
                r = rf[(rf["region"] == reg) & (rf["level"] == lev)].sort_values("date")
                ax.plot(_doy_axis(r["date"]), r["ref_gmean"], color="black",
                        lw=1.8, ls=ystyle[yr], alpha=0.9, zorder=3)
            for s in sources:
                r = rolls[s.run]
                r = r[(r["region"] == reg) & (r["level"] == lev)].copy()
                if r.empty:
                    continue
                r["valid_time"] = r["init_date"] + pd.to_timedelta(r["lead_hours"], unit="h")
                r["lead_day_idx"] = (r["lead_hours"] // 24).astype(int)
                for d in sorted(r["init_date"].unique()):
                    g = r[r["init_date"] == d]
                    daily = (g.groupby("lead_day_idx")
                               .agg(vt=("valid_time", "mean"), val=("pred_gmean", "mean"))
                               .reset_index())
                    ax.plot(_doy_axis(daily["vt"]), daily["val"], color=s.color,
                            lw=0.5, ls=ystyle[s.year], alpha=0.45, zorder=2)
            mh = [Line2D([], [], color=S.model_color(m), lw=2,
                         label=S.MODEL_PRETTY.get(m, m))
                  for m in dict.fromkeys(s.model for s in sources)]
            yh = ([Line2D([], [], color="black", lw=1.8, ls="-", label="reference (truth)")]
                  + [Line2D([], [], color="0.35", lw=1.8, ls=ystyle[y], label=str(y))
                     for y in years])
            l1 = ax.legend(handles=mh, title="model", loc="upper left",
                           fontsize=8, framealpha=0.9)
            ax.add_artist(l1)
            ax.legend(handles=yh, title="year (line style)", loc="upper right",
                      fontsize=8, framealpha=0.9)
            ax.set_title(f"{area.capitalize()}-mean {label} at {lev} hPa — "
                         f"multi-year rollouts ({', '.join(str(y) for y in years)})")
            ax.set_ylabel(f"{label} @{lev}hPa {area} mean [{units}]")
            ax.set_xlabel("Day of year")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.margins(x=0.01); ax.grid(alpha=0.25)
            fig.tight_layout()
            figdir = S.figure_dir(sources, 0, reg, var, "spaghetti")
            out = figdir / fig_naming.figure_name(
                S.model_token(sources), sources[0].dataset, reg, var, lev,
                "-".join(str(y) for y in years), "multiyear", "spaghetti", ext="pdf")
            fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  wrote {out.name}")


def main():
    sources = S.resolve_sources()
    levels = S.requested_levels(sources)
    regions = C.selected_regions()
    if len({s.year for s in sources}) > 1:
        for var in C.selected_variables():
            plot_variable_multiyear(sources, var, levels, regions)
    else:
        periods = C.selected_periods()
        for var in C.selected_variables():
            plot_variable(sources, var, levels, regions, periods)
    print(f"done -> {C.FIG_ROOT}/{S.run_label(sources)}/<period>/<region>/<variable>/spaghetti/")


if __name__ == "__main__":
    main()
