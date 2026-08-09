#!/usr/bin/env python
"""Case study: NeuralGCM rollouts over EUROPE heat-wave episodes (mixture / p99).

For each detected episode (see ``heatwave_mask``) this renders, using ONLY the grid
cells in the episode's heat-wave FOOTPRINT (a ``GridPoints`` set):

  Lagrangian (follow the afflicted region over its heat-wave lifetime + rollout
  window before + brief window after):
    * spaghetti   -- footprint-area-mean field, ERA5 (thick black) + one thin line
                     per rollout init in [start-10d, end], over valid [start-10d,
                     end+10d]; the episode span is shaded.
    * rmse-vs-lead-- RMSE + bias vs lead over the footprint, aggregated over the
                     rollouts initialised in [start-10d, end] (the forecasts rolling
                     into the event).

  Eulerian (the whole footprint analysed over the episode's day-10 valid window;
  cells never in the heat wave are left NaN / uncoloured):
    * drift map   -- ERA5 mean | forecast day-10 mean | mean day-10 error, cropped
                     to Europe, coloured only on the footprint.

Variables: temperature@850 hPa and geopotential@500 hPa (env EVAL_VARS / CS_LEVELS
override). The definition + percentile come from ``HW_PCT`` (=0.99) + the mixture
definition. Figures: ``outputs/figures/case_study/<def>_<ptag>/<year>/<episode>/``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from aic import config
from aic.controller.eval import eval_common as C
from aic.controller.eval import gridpoints as GP
from aic.controller.eval import sources as S
from aic.view import plotting as P
from aic.controller.casestudy import heatwave_mask as HM
from aic.controller.heatwave import definitions as D
from aic.regions import region_extent

BEFORE = config.env_int("CS_BEFORE", 10)    # rollout window before episode
AFTER = config.env_int("CS_AFTER", 10)      # brief window after (valid axis)
# variable -> level(s) to evaluate (T at 850, Z at 500 for the case study)
CS_LEVELS = {"temperature": [850], "geopotential": [500]}


# --------------------------------------------------------------------------- #
# episode -> grid-point set + rollout file window
# --------------------------------------------------------------------------- #
def footprint_points(source: S.Source, ep: HM.Episode, year: int) -> GP.GridPoints:
    """The episode's footprint as a GridPoints on the source's prediction grid.

    The footprint is detected on the 2.8deg grid; nearest-neighbour reindexing onto
    the source grid makes it align by coordinates with that source's rollouts/truth
    (identity for the NeuralGCM 2.8deg grid, an upsample for GraphCast's 0.25deg)."""
    lat, lon = source.prediction_grid()
    m = ep.footprint.reindex(latitude=lat, longitude=lon, method="nearest")
    return GP.GridPoints.from_mask(f"hw_{year}_{ep.tag}", m, region_extent("europe"))


def episode_files(source: S.Source, ep: HM.Episode, before: int, after: int):
    """Rollout pred files whose INIT date lies in [start-before, end+after]."""
    lo = (ep.start - pd.Timedelta(days=before)).date()
    hi = (ep.end + pd.Timedelta(days=after)).date()
    out = []
    for f in source.pred_files():
        d = pd.to_datetime(f.stem.replace(f"pred_{source.year}_", "")).date()
        if lo <= d <= hi:
            out.append(f)
    return out


def fig_dir(defn, year: int, ep: HM.Episode) -> Path:
    d = (C.FIG_ROOT / "case_study" / f"{defn.name}_{D.PTAG}" / str(year)
         / f"{ep.tag}_{ep.start:%Y-%m-%d}_{ep.end:%Y-%m-%d}")
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Lagrangian: spaghetti over the episode window (footprint area-mean)
# --------------------------------------------------------------------------- #
def spaghetti_episode(sources, defn, year, ep, var, lev):
    meta = C.VARIABLES[var]
    label, units = meta["label"], meta["units"]
    v0 = ep.start - pd.Timedelta(days=BEFORE)
    v1 = ep.end + pd.Timedelta(days=AFTER)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axvspan(ep.start, ep.end, color="#f2c14e", alpha=0.25, zorder=0,
               label="heat-wave episode")
    drew = False
    for src in sources:
        gp = footprint_points(src, ep, year)
        files = episode_files(src, ep, BEFORE, 0)    # inits rolling into the event
        if not files:
            print(f"  [spaghetti] {ep.tag} {var} {src.model}: no rollouts; skip")
            continue
        roll = src.rollout_gmean(var, meta["short"], [lev], [gp],
                                 files=files, cache=False)
        P.draw_rollout_bundle(ax, roll, src.color, lw=0.7, alpha=0.55)
        ax.plot([], [], color=src.color, lw=1.6, label=f"{src.pretty} 10-day rollout")
        drew = True
    if not drew:
        plt.close(fig); return

    # ERA5 reference: footprint-area-mean, daily, over the valid window (from src 0)
    ref_src = sources[0]
    truth = ref_src.truth_at_levels(var, [lev])
    ref = GP.masked_area_mean(truth.sel(level=lev), footprint_points(ref_src, ep, year))
    ref = ref.to_dataframe(name="ref").reset_index()
    ref["date"] = pd.to_datetime(ref["time"]).dt.floor("D")
    ref = ref.groupby("date", as_index=False)["ref"].mean()
    ref = ref[(ref["date"] >= v0) & (ref["date"] <= v1)]
    ax.plot(ref["date"], ref["ref"], color="black", lw=2.2, zorder=3,
            label=f"{ref_src.ref_label} (daily mean)")
    ax.set_title(f"Europe heat-wave {ep.label} — footprint-mean {label} @ {lev} hPa "
                 f"({defn.name}, > {D.PTAG})", fontsize=12, color=P.INK, loc="left")
    ax.set_ylabel(f"{label} @{lev}hPa footprint mean [{units}]", color=P.INK)
    ax.set_xlabel("Valid time", color=P.INK)
    ax.set_xlim(v0, v1)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.25); ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = fig_dir(defn, year, ep) / f"spaghetti_{meta['short']}_L{lev:04d}.pdf"
    written = P.save_fig(fig, out)
    print(f"  wrote {', '.join(p.name for p in written)}")


# --------------------------------------------------------------------------- #
# Lagrangian: RMSE + bias vs lead over the footprint (episode rollouts)
# --------------------------------------------------------------------------- #
def skill_episode(sources, defn, year, ep, var, lev):
    """Two SEPARATE Lagrangian figures over the footprint (kept apart for
    readability): RMSE vs lead, and mean bias vs lead. One line per model,
    coloured by the shared model palette."""
    meta = C.VARIABLES[var]
    label, units = meta["label"], meta["units"]
    curves = []            # list of (source, aggregated sorted dataframe)
    raw = []               # list of (model, per-init drift df) -> pooled aggregate
    for src in sources:
        gp = footprint_points(src, ep, year)
        files = episode_files(src, ep, BEFORE, 0)
        if not files:
            print(f"  [skill] {ep.tag} {var} {src.model}: no rollouts; skip")
            continue
        per = src.drift_per_init(var, meta["short"], [lev], [gp],
                                 files=files, cache=False)
        raw.append((src.model, per))
        agg = C.aggregate(per, ci_metrics=("bias", "rmse"))
        a = agg[(agg["region"] == gp.key) & (agg["level"] == lev)].sort_values("lead_hours")
        if not a.empty:
            curves.append((src, a))
    if not curves:
        return raw
    lines = [(src.color, src.pretty, a) for src, a in curves]
    for metric, ylab, zero in [("rmse", f"RMSE [{units}]", False),
                               ("bias", f"mean bias [{units}]", True)]:
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        P.draw_skill_metric(ax, lines, metric, zero_line=zero)
        ax.set_ylabel(P.skill_ylabel(metric, label, lev, units))
        if len(curves) > 1:
            ax.legend(loc="best", fontsize=9, framealpha=0.9)
        P.despine(ax)
        fig.tight_layout()
        out = fig_dir(defn, year, ep) / f"{metric}-vs-lead_{meta['short']}_L{lev:04d}.pdf"
        written = P.save_fig(fig, out)
        print(f"  wrote {', '.join(p.name for p in written)}")
    return raw


# --------------------------------------------------------------------------- #
# Eulerian: mean day-10 error map over the footprint (Europe extent)
# --------------------------------------------------------------------------- #
def _episode_day10(source, ep, year, var, lev):
    """(fc_day10_mean, ref_mean, drift, n_steps, gp) footprint-masked for one
    source over the episode's day-10 valid window, or None if no rollouts."""
    gp = footprint_points(source, ep, year)
    files = episode_files(source, ep, BEFORE, 0)
    if not files:
        return None
    truth = source.truth_at_levels(var, [lev]).sel(level=lev)
    facc = None; racc = None; n = 0
    for f in files:
        ds = source.open_pred(f)
        day10 = (S.day10_slab(ds, source.regrid_field(ds[var].sel(level=lev)))
                 .transpose("time", "latitude", "longitude"))
        if day10.sizes["time"] == 0:
            ds.close(); continue
        vt = S.day10_slab(ds, ds["valid_time"]).values
        tru = (truth.sel(time=vt, method="nearest")
               .transpose("time", "latitude", "longitude"))
        fsum = day10.sum("time")
        rsum = tru.sum("time").assign_coords(latitude=fsum.latitude,
                                             longitude=fsum.longitude)
        facc = fsum if facc is None else facc + fsum
        racc = rsum if racc is None else racc + rsum
        n += day10.sizes["time"]
        ds.close()
    if not n:
        return None
    fpm = gp.mask
    fc = (facc / n).where(fpm); rf = (racc / n).where(fpm)
    return fc, rf, (fc - rf), n, gp


def driftmap_episode(sources, defn, year, ep, var, lev):
    """Eulerian day-10 maps over the footprint, all models SIDE BY SIDE in one
    figure: shared ERA5 panel + (day-10 mean, mean day-10 error) per model."""
    meta = C.VARIABLES[var]
    label, units, fcmap = meta["label"], meta["units"], meta["cmap"]
    got = [(s, _episode_day10(s, ep, year, var, lev)) for s in sources]
    got = [(s, r) for s, r in got if r is not None]
    if not got:
        print(f"  [driftmap] {ep.tag} {var}: no rollouts; skip")
        return
    extent = region_extent("europe")
    field_arrays = [r[0] for _, r in got] + [got[0][1][1]]    # day-10 means + ref
    vmin, vmax, dlim = P.map_scales(field_arrays, [r[2] for _, r in got])

    npan = 1 + 2 * len(got)
    fig, axes = plt.subplots(1, npan, figsize=(5.4 * npan, 4.6), squeeze=False)
    axes = axes[0]
    ref_src = got[0][0]
    P.map_panel(axes[0], got[0][1][1], cmap=fcmap, vmin=vmin, vmax=vmax,
                title=f"{ref_src.ref_label} mean", cbar_label=f"{label} [{units}]",
                extent=extent, fig=fig, coast=P.draw_coastlines)
    col = 1
    for src, (fc, rf, dr, n, gp) in got:
        gm = float(GP.masked_area_mean(dr, gp))
        P.map_panel(axes[col], fc, cmap=fcmap, vmin=vmin, vmax=vmax,
                    title=f"{src.pretty} day-10 mean", cbar_label=f"{label} [{units}]",
                    extent=extent, fig=fig, coast=P.draw_coastlines)
        P.map_panel(axes[col + 1], dr, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                    title=f"{src.pretty} day-10 error\n(footprint mean {gm:+.4g} {units})",
                    cbar_label=f"error [{units}]", extent=extent, fig=fig,
                    coast=P.draw_coastlines)
        col += 2
    models = " vs ".join(dict.fromkeys(s.pretty for s, _ in got))
    fig.suptitle(f"Europe heat-wave {ep.label} — day-10 {label}@{lev} hPa over the "
                 f"heat-wave footprint ({ep.n_cells} cells, {models}, "
                 f"{defn.name} > {D.PTAG})", y=1.03, fontsize=12.5)
    fig.tight_layout()
    out = fig_dir(defn, year, ep) / f"drift-map_{meta['short']}_L{lev:04d}.png"
    written = P.save_fig(fig, out)
    print(f"  wrote {', '.join(p.name for p in written)}")


# --------------------------------------------------------------------------- #
# per-year overview: coverage timeline with episodes shaded
# --------------------------------------------------------------------------- #
def year_overview(defn, year, active_da, eps, region="europe"):
    cov = HM.region_coverage(active_da, region)
    times = pd.to_datetime(active_da["time"].values)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(times, 100 * cov, color="#b0413e", lw=1.3)
    ax.fill_between(times, 0, 100 * cov, color="#d98880", alpha=0.35)
    for e in eps:
        ax.axvspan(e.start, e.end, color="#f2c14e", alpha=0.3, zorder=0)
        ax.text(e.start, ax.get_ylim()[1] * 0.92, e.tag, fontsize=7, color="#666")
    ax.set_title(f"Europe heat-wave coverage in {year} ({defn.name} > {D.PTAG}, "
                 f"{len(eps)} episodes)", fontsize=12, color=P.INK, loc="left")
    ax.set_ylabel("% of Europe in heat wave", color=P.INK)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(times[0], times[-1]); ax.margins(x=0)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    d = C.FIG_ROOT / "case_study" / f"{defn.name}_{D.PTAG}" / str(year)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"_overview_coverage_{year}.pdf"
    fig.tight_layout(); P.save_fig(fig, out)
    print(f"[overview] wrote {out}")


# --------------------------------------------------------------------------- #
def run_year(sources, defn, year, pool=None, window=HM.DEFAULT_WINDOW, region="europe"):
    models = "+".join(s.model for s in sources)
    print(f"=== case study {defn.name} > {D.PTAG}: {models} ({year}) ===", flush=True)
    active = HM.active_mask_da(defn, year, window, region)   # 2.8deg, grid-independent
    eps = HM.episodes(active, region)
    only = config.env_str("CS_ONLY_EP", "").strip()
    if only:
        want = {int(x) for x in only.replace(",", " ").split()}
        eps = [e for e in eps if e.idx in want]
    print(f"[{year}] {len(eps)} episode(s): "
          + ", ".join(f"{e.tag}({e.label},{e.n_cells}c)" for e in eps), flush=True)
    if not only:
        year_overview(defn, year, active, eps, region)
    avail = set.intersection(*(set(s.prediction_levels()) for s in sources))
    variables = C.selected_variables()
    for ep in eps:
        print(f"-- {ep.tag} {ep.label} ({ep.n_days}d, {ep.n_cells} cells) --", flush=True)
        for var in variables:
            for lev in CS_LEVELS.get(var, [850]):
                if lev not in avail:
                    continue
                spaghetti_episode(sources, defn, year, ep, var, lev)
                raw = skill_episode(sources, defn, year, ep, var, lev) or []
                if pool is not None:
                    for model, per in raw:
                        pool[(model, var, lev)].append(per)
                driftmap_episode(sources, defn, year, ep, var, lev)
    return len(eps)


def _aggregate_curves(pool, model_sources, var, lev):
    """[(color, pretty, aggregated sorted df)] for the models present in ``pool``,
    pooling every heat wave in the pool together (region -> 'all')."""
    curves = []
    for src in model_sources:
        pers = pool.get((src.model, var, lev), [])
        if not pers:
            continue
        allper = pd.concat(pers, ignore_index=True)
        allper["region"] = "all"
        curves.append((src.color, src.pretty,
                       C.aggregate(allper, ci_metrics=("bias", "rmse")).sort_values("lead_hours")))
    return curves


def shared_ylims(pools_sources):
    """Per (var, level, metric) y-axis range spanning EVERY (pool, model) curve, so the
    per-year _aggregate plots (and the all-years one) share one scale and the years are
    directly comparable. ``pools_sources`` is a list of (pool, model_sources). Bias
    ranges always include 0 (the zero line is drawn); a 5% pad is added."""
    lims = {}
    for var in C.selected_variables():
        for lev in CS_LEVELS.get(var, [850]):
            for metric in ("rmse", "bias"):
                mn, mx = [], []
                for pool, srcs in pools_sources:
                    for _, _, agg in _aggregate_curves(pool, srcs, var, lev):
                        # span the CI band too (if present) so it isn't clipped
                        lo_c = agg[f"{metric}_lo"] if f"{metric}_lo" in agg else agg[metric]
                        hi_c = agg[f"{metric}_hi"] if f"{metric}_hi" in agg else agg[metric]
                        mn.append(float(lo_c.min()))
                        mx.append(float(hi_c.max()))
                if not mn:
                    continue
                lo, hi = min(mn), max(mx)
                if metric == "bias":
                    lo, hi = min(lo, 0.0), max(hi, 0.0)
                pad = 0.05 * (hi - lo) if hi > lo else (abs(hi) or 1.0) * 0.05
                lims[(var, lev, metric)] = (lo - pad, hi + pad)
    return lims


def run_aggregate(pool, defn, model_sources, n_events, subdir, scope_label,
                  ylims=None, fmts=None):
    """Cross-episode comparison: pool the per-init drift over a SET of heat waves and
    plot the mean RMSE and mean bias vs lead per model -- the models' average skill
    during those heat waves. Only RMSE + bias (no regional maps or spaghetti).

    ``subdir`` is the output folder under case_study/<def>_<ptag>/ and ``scope_label``
    names the set in the title. Called twice: once per year (``<year>/_aggregate/``,
    that year's episodes) and once for everything (``_aggregate/``, all years)."""
    outdir = C.FIG_ROOT / "case_study" / f"{defn.name}_{D.PTAG}" / subdir
    outdir.mkdir(parents=True, exist_ok=True)
    for var in C.selected_variables():
        meta = C.VARIABLES[var]
        short, units, label = meta["short"], meta["units"], meta["label"]
        for lev in CS_LEVELS.get(var, [850]):
            curves = _aggregate_curves(pool, model_sources, var, lev)
            if not curves:
                continue
            for metric, ylab, zero in [("rmse", f"RMSE [{units}]", False),
                                       ("bias", f"mean bias [{units}]", True)]:
                fig, ax = plt.subplots(figsize=(6.8, 4.4))
                P.draw_skill_metric(ax, curves, metric, zero_line=zero)
                if ylims and (var, lev, metric) in ylims:
                    ax.set_ylim(*ylims[(var, lev, metric)])   # shared scale across years
                ax.set_ylabel(P.skill_ylabel(metric, label, lev, units))
                if len(curves) > 1:
                    ax.legend(loc="best", fontsize=9, framealpha=0.9)
                P.despine(ax)
                fig.tight_layout()
                out = outdir / f"{metric}-vs-lead_{short}_L{lev:04d}.pdf"
                for p in P.save_fig(fig, out, fmts=fmts):
                    print(f"[aggregate] wrote {p.relative_to(C.FIG_ROOT)}", flush=True)


def run_aggregate_byyear(year_runs, defn, ylims=None, fmts=None):
    """Combined figure per metric: the years (2023 | 2026) side by side, sharing one
    y-axis, each panel pooling that year's heat waves -- complements the individual
    <year>/_aggregate plots. Written to case_study/<def>_<ptag>/_by_year/."""
    outdir = C.FIG_ROOT / "case_study" / f"{defn.name}_{D.PTAG}" / "_by_year"
    outdir.mkdir(parents=True, exist_ok=True)
    for var in C.selected_variables():
        meta = C.VARIABLES[var]
        short, units, label = meta["short"], meta["units"], meta["label"]
        for lev in CS_LEVELS.get(var, [850]):
            for metric, zero in [("rmse", False), ("bias", True)]:
                panels = [(str(year), _aggregate_curves(ypool, srcs, var, lev))
                          for year, ypool, srcs, _ in year_runs]
                panels = [(y, c) for y, c in panels if c]
                if not panels:
                    continue
                fig = P.skill_facets(panels, metric, zero_line=zero,
                                     ylabel=P.skill_ylabel(metric, label, lev, units),
                                     ylim=(ylims or {}).get((var, lev, metric)))
                out = outdir / f"{metric}-vs-lead_{short}_L{lev:04d}.pdf"
                for p in P.save_fig(fig, out, fmts=fmts):
                    print(f"[by-year] wrote {p.relative_to(C.FIG_ROOT)}", flush=True)


def main():
    from collections import defaultdict
    defn = D.BY_NAME[config.env_str("HW_CS_DEF", "mixture")]
    years = [int(y) for y in config.env_list("CS_YEARS", ["2023", "2026"])]
    dataset = config.env_str("EVAL_DATASET", "era5")
    models = config.env_list("CS_MODELS", ["neuralgcm", "graphcast"])
    pool = defaultdict(list)             # every episode of every year (all-years agg)
    rep, n_events = {}, 0
    year_runs = []                       # (year, ypool, srcs, n_year), in order
    for year in years:
        srcs = []
        for m in models:
            run = S.run_for(m, dataset, year)
            if run is None:
                print(f"[cs] no run for {m}/{dataset}/{year}; skip", flush=True)
                continue
            s = S.Source.from_run(run, color=S.model_color(m))
            srcs.append(s); rep[m] = s
        if not srcs:
            continue
        ypool = defaultdict(list)         # this year's episodes only (yearly agg)
        n_year = run_year(srcs, defn, year, pool=ypool)
        n_events += n_year
        for k, v in ypool.items():        # roll the year into the all-years pool
            pool[k].extend(v)
        year_runs.append((year, ypool, srcs, n_year))

    # one shared y-scale per (var, level, metric) across all years + the all-years
    # pool, so the per-year _aggregate plots are directly comparable between years.
    all_srcs = [rep[m] for m in models if m in rep]
    pools_sources = [(yp, s) for _, yp, s, _ in year_runs]
    if all_srcs:
        pools_sources.append((pool, all_srcs))
    ylims = shared_ylims(pools_sources)
    # file type(s) for the aggregate figures (default pdf); e.g. CS_AGG_FMTS="pdf png"
    fmts = config.env_list("CS_AGG_FMTS") or None

    for year, ypool, srcs, n_year in year_runs:
        run_aggregate(ypool, defn, srcs, n_year, subdir=f"{year}/_aggregate",
                      scope_label=f"the {n_year} {year} {defn.name} heat waves",
                      ylims=ylims, fmts=fmts)
    if all_srcs:
        run_aggregate(pool, defn, all_srcs, n_events,
                      subdir="_aggregate", scope_label=f"all {defn.name} heat waves",
                      ylims=ylims, fmts=fmts)
    if year_runs:
        run_aggregate_byyear(year_runs, defn, ylims=ylims, fmts=fmts)
    print(f"done -> {C.FIG_ROOT}/case_study/{defn.name}_{D.PTAG}/ ({n_events} events)")


if __name__ == "__main__":
    main()
