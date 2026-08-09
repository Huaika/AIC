#!/usr/bin/env python
"""Out-of-distribution (OOD) analysis: NeuralGCM vs GraphCast across CLIMATES.

Overlays several (model, year) rollouts to see how forecast skill and the annual
cycle shift between the historical 1955, present-day 2023 and the NextGEMS-2049
future-climate runs. Two figure families, all under a dedicated output folder
``outputs/figures/out_of_distribution/``:

  skill    -- per YEAR, RMSE (one figure) and mean bias (one figure) vs lead, with
              NeuralGCM and GraphCast overlaid (colour = model).
  spaghetti-- a SINGLE figure overlaying every (model, year) rollout bundle plus one
              reference (truth) line per (dataset, year), on a day-of-year x-axis
              (colour = model, line style = year).

Sources come from ``EVAL_RUNS`` (explicit run-key list, any model/dataset/year
mix), e.g.
    EVAL_RUNS=era5_1955,era5_2023,nextgems2049,\\
              graphcast_era5_1955,graphcast_era5_2023,graphcast_nextgems_2049 \\
    EVAL_VARS=temperature NG_LEVELS=850 EVAL_REGIONS=world  python -m aic.view.ood
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from aic import config
from aic.controller.eval import eval_common as C
from aic.controller.eval import sources as S
from aic.view import naming as fig_naming
from aic.view import plotting as P
from aic.view.spaghetti import build_ref

OOD_ROOT = C.FIG_ROOT / "out_of_distribution"
# output file type(s) for the OOD figures (default pdf); e.g. OOD_FMTS="pdf png"
OOD_FMTS = config.env_list("OOD_FMTS") or None


def _figdir(*parts):
    d = OOD_ROOT.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Per-year skill: RMSE and bias vs lead, both models overlaid
# --------------------------------------------------------------------------- #
_METRICS = [("rmse", False, "drift_rmse"), ("bias", True, "drift_bias")]


def _year_aggs(srcs, var, short, levels, region):
    """[(source, aggregated df: mean bias/rmse + CI over leads/levels)] for one year."""
    return [(s, C.aggregate(s.drift_per_init(var, short, levels, [region]),
                            ci_metrics=("bias", "rmse"))) for s in srcs]


def _curves_at(year_aggs, region, lev):
    """[(color, pretty, sorted df)] for one (region, level) from [(source, agg)]."""
    out = []
    for s, agg in year_aggs:
        a = agg[(agg["region"] == region) & (agg["level"] == lev)].sort_values("lead_hours")
        if not a.empty:
            out.append((s.color, s.pretty, a))
    return out


def skill_year(year, year_aggs, var, levels, region):
    """One RMSE figure + one bias figure for a single year, overlaying every model."""
    meta = C.VARIABLES[var]
    units, label = meta["units"], meta["label"]
    area = "global" if region == "world" else region
    srcs = [s for s, _ in year_aggs]
    for lev in C.render_levels(levels):
        curves = _curves_at(year_aggs, region, lev)
        if not curves:
            print(f"  [skill] {year} {var}@{lev}: no data; skip")
            continue
        n_ref = int(curves[0][2]["n_init"].iloc[0])
        models = " vs ".join(dict.fromkeys(c[1] for c in curves))
        for metric, zero, kind in _METRICS:
            fig, ax = plt.subplots(figsize=(6.6, 4.4))
            P.draw_skill_metric(ax, curves, metric, zero_line=zero)
            ax.set_title(f"{models} — {year}, {lev} hPa {label} ({area}, "
                         f"mean of {n_ref} inits)", fontsize=11)
            ax.set_ylabel(P.skill_ylabel(metric, label, lev, units))
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
            P.despine(ax)
            fig.tight_layout()
            out = _figdir(var, kind) / fig_naming.figure_name(
                S.model_token(srcs), srcs[0].dataset, region, var, lev, year,
                "entire-year", kind, ext="pdf")
            for p in P.save_fig(fig, out, fmts=OOD_FMTS):
                print(f"  wrote {p.relative_to(OOD_ROOT)}")


def skill_facets_years(aggs_by_year, var, levels, region):
    """Combined figure per metric: the years (1955 | 2023 | 2049) side by side, sharing
    one y-axis, so the shift in skill across climates reads at a glance."""
    meta = C.VARIABLES[var]
    units, label = meta["units"], meta["label"]
    area = "global" if region == "world" else region
    years = sorted(aggs_by_year)
    any_srcs = [s for yr in years for s, _ in aggs_by_year[yr]]
    for lev in C.render_levels(levels):
        for metric, zero, kind in _METRICS:
            panels = [(yr, _curves_at(aggs_by_year[yr], region, lev)) for yr in years]
            panels = [(yr, c) for yr, c in panels if c]
            if not panels:
                continue
            models = " vs ".join(dict.fromkeys(c[1] for _, cs in panels for c in cs))
            fig = P.skill_facets(panels, metric, zero_line=zero,
                                 ylabel=P.skill_ylabel(metric, label, lev, units))
            fig.suptitle(f"{models} — {label}@{lev} hPa ({area})", y=1.02, fontsize=12)
            out = _figdir(var, kind) / fig_naming.figure_name(
                S.model_token(any_srcs), any_srcs[0].dataset, region, var, lev,
                "-".join(str(y) for y in years), "by-year", kind, ext="pdf")
            for p in P.save_fig(fig, out, fmts=OOD_FMTS):
                print(f"  wrote {p.relative_to(OOD_ROOT)}")


# --------------------------------------------------------------------------- #
# Combined multi-year spaghetti: every (model, year) on one day-of-year axis
# --------------------------------------------------------------------------- #
def spaghetti_multiyear(sources, var, levels, region):
    meta = C.VARIABLES[var]
    short, units, label = meta["short"], meta["units"], meta["label"]
    area = "global" if region == "world" else region
    years = sorted({s.year for s in sources})
    rolls = {s.run: s.rollout_gmean(var, short, levels, [region]) for s in sources}
    refs = {}                                          # (dataset, year) -> frame
    for s in sources:
        key = (s.dataset, s.year)
        if key not in refs:
            refs[key] = build_ref(s, var, levels, [region])

    for lev in C.render_levels(levels):
        fig, ax = plt.subplots(figsize=(13, 5.4))
        # every line is solid; the years are distinguished by a label written to the
        # right of the plot at the height of each truth line's final value.
        for (dset, yr), rf in refs.items():
            r = rf[(rf["region"] == region) & (rf["level"] == lev)].sort_values("date")
            x = P.doy_axis(r["date"])
            ax.plot(x, r["ref_gmean"], color=S.REF_COLOR, lw=1.8, alpha=0.9, zorder=3)
            ax.annotate(str(yr), xy=(x[-1], float(r["ref_gmean"].iloc[-1])),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=9, fontweight="bold", color=P.INK,
                        annotation_clip=False)
        for s in sources:
            r = rolls[s.run]
            r = r[(r["region"] == region) & (r["level"] == lev)]
            if r.empty:
                continue
            P.draw_rollout_bundle(ax, r, s.color, lw=0.5, alpha=0.45,
                                  x_transform=P.doy_axis)
        mh = ([Line2D([], [], color=S.REF_COLOR, lw=1.8, label="reference (truth)")]
              + [Line2D([], [], color=S.model_color(m), lw=2,
                        label=S.MODEL_PRETTY.get(m, m))
                 for m in dict.fromkeys(s.model for s in sources)])
        ax.legend(handles=mh, title="model", loc="upper left",
                  fontsize=8, framealpha=0.9)
        ax.set_title(f"{area.capitalize()}-mean {label} at {lev} hPa — "
                     f"out-of-distribution rollouts ({', '.join(str(y) for y in years)})")
        ax.set_ylabel(f"{label} @{lev}hPa {area} mean [{units}]")
        ax.set_xlabel("Day of year")
        P.month_axis(ax); ax.margins(x=0.01); ax.grid(alpha=0.25)
        P.despine(ax)
        fig.tight_layout()
        out = _figdir(var, "spaghetti") / fig_naming.figure_name(
            S.model_token(sources), sources[0].dataset, region, var, lev,
            "-".join(str(y) for y in years), "multiyear", "spaghetti", ext="pdf")
        for p in P.save_fig(fig, out, fmts=OOD_FMTS):
            print(f"  wrote {p.relative_to(OOD_ROOT)}")


def main():
    sources = S.resolve_sources()
    if len(sources) < 2:
        raise SystemExit("OOD analysis needs several runs; set EVAL_RUNS to a list.")
    levels = S.requested_levels(sources)
    regions = C.selected_regions()
    for var in C.selected_variables():
        short = C.VARIABLES[var]["short"]
        by_year = {}
        for s in sources:
            by_year.setdefault(s.year, []).append(s)
        for region in regions:
            print(f"=== OOD skill: {var} / {region} ===")
            aggs = {yr: _year_aggs(srcs, var, short, levels, region)
                    for yr, srcs in by_year.items()}
            for yr in sorted(aggs):
                skill_year(yr, aggs[yr], var, levels, region)
            print(f"=== OOD skill (years side by side): {var} / {region} ===")
            skill_facets_years(aggs, var, levels, region)
            print(f"=== OOD spaghetti: {var} / {region} ===")
            spaghetti_multiyear(sources, var, levels, region)
    print(f"done -> {OOD_ROOT}/<variable>/{{drift_rmse,drift_bias,spaghetti}}/")


if __name__ == "__main__":
    main()
