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

import matplotlib.pyplot as plt

from aic.controller.eval import eval_common as C
from aic.controller.eval import sources as S
from aic.view import naming as fig_naming
from aic.view import plotting as P


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
                aggs[s.run] = C.aggregate(sub, ci_metrics=("bias", "rmse"))
        if not aggs:
            print(f"  [skip] no init-days in {C.period_dir_name(period)}")
            continue
        for reg in regions:
            figdir = S.figure_dir(sources, period, reg, var, "drift_stats")
            area = "global" if reg == "world" else reg
            for lev in C.render_levels(levels):
                # RMSE and bias are drawn as SEPARATE figures (readability); each
                # overlays one line per source, coloured by the shared model palette.
                title_model = sources[0].ref_label if single else \
                    " vs ".join(dict.fromkeys(s.pretty for s in sources))
                # gather each source's (colour, label, curve) for this region/level
                curves, n_ref = [], None
                for s in sources:
                    if s.run not in aggs:
                        continue
                    ar = aggs[s.run]
                    a = ar[(ar["region"] == reg) & (ar["level"] == lev)] \
                        .sort_values("lead_hours")
                    if a.empty:
                        continue
                    n_ref = int(a["n_init"].iloc[0])
                    curves.append((s.color, s.pretty, a))
                if n_ref is None:
                    continue
                for metric, ylab, zero, kind in [
                        ("rmse", f"RMSE [{units}]", False, "drift_rmse"),
                        ("bias", f"mean bias [{units}]", True, "drift_bias")]:
                    fig, ax = plt.subplots(figsize=(6.5, 4.4))
                    P.draw_skill_metric(ax, curves, metric, zero_line=zero, lw=1.8)
                    ax.set_title(f"{title_model} — {lev} hPa {label} ({area}, "
                                 f"mean of {n_ref} daily inits)")
                    ax.set_ylabel(P.skill_ylabel(metric, label, lev, units))
                    P.despine(ax)
                    if not single:
                        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
                    fig.tight_layout()
                    out = figdir / fig_naming.figure_name(
                        S.model_token(sources), sources[0].dataset, reg, var, lev,
                        sources[0].year, fig_naming.months_token(period), kind, ext="pdf")
                    P.save_fig(fig, out)
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
