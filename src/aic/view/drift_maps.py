#!/usr/bin/env python
"""10-day drift maps (Rackow et al. 2024 Fig. 3 style) -- multi-source aware.

drift = annual-(or month-)mean of the model's day-10 fields minus the reference
mean:
  fc_day10_clim = mean over inits of the rollout's final-day mean (216..240 h)
  ref_clim      = reference mean field
  drift         = fc_day10_clim - ref_clim   (model grid)

Single source: three panels (forecast day-10 clim | reference | drift) --
byte-identical name/paths to before. Multiple sources
(EVAL_SOURCES=neuralgcm,graphcast + EVAL_YEAR): one forecast panel per source +
one shared reference panel + one drift panel per source, i.e.
  [ NGCM fc | GC fc | ERA5 truth | NGCM drift | GC drift ]
Each source uses its OWN grid (NeuralGCM 2.8deg / GraphCast 0.25deg), so panels
may differ in resolution; the reference panel is drawn from the first source.

Output: figures/<run-or-compare>/<period>/<region>/<variable>/drift_maps/.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from aic.controller.eval import eval_common as C
from aic.controller.eval import sources as S
from aic.view import naming as fig_naming
from aic.view import plotting as P


def plot_period_region(fields, sources, var, short, units, label, fcmap,
                       levels, period, reg):
    """fields: {run: xr.Dataset(fc_day10_clim, ref_clim, drift)} per source."""
    figdir = S.figure_dir(sources, period, reg, var, "drift_maps")
    extent = C.region_extent(reg)
    area = "global" if reg == "world" else reg
    ref_src = sources[0]
    for lev in levels:
        fcs = {r: C.select_region(fields[r]["fc_day10_clim"].sel(level=lev), reg)
               for r in fields}
        drs = {r: C.select_region(fields[r]["drift"].sel(level=lev), reg)
               for r in fields}
        rf = C.select_region(fields[ref_src.run]["ref_clim"].sel(level=lev), reg)

        # shared field colour scale over all forecasts + the reference; symmetric
        # drift scale over all drift panels
        vmin, vmax, dlim = P.map_scales(list(fcs.values()) + [rf], list(drs.values()))

        npan = 2 * len(sources) + 1
        fig, axes = plt.subplots(1, npan, figsize=(6.3 * npan, 4.3), squeeze=False)
        axes = axes[0]
        col = 0
        for src in sources:  # forecast panels
            ninit = int(fields[src.run].attrs.get("n_inits", 0))
            P.map_panel(axes[col], fcs[src.run], cmap=fcmap, vmin=vmin, vmax=vmax,
                        title=f"{src.pretty} day-10 climatology\n(mean of {ninit} forecasts)",
                        cbar_label=f"{label} [{units}]", extent=extent, fig=fig,
                        coast=P.draw_coastlines)
            col += 1
        P.map_panel(axes[col], rf, cmap=fcmap, vmin=vmin, vmax=vmax,  # shared ref
                    title=f"{ref_src.ref_label} reference mean",
                    cbar_label=f"{label} [{units}]", extent=extent, fig=fig,
                    coast=P.draw_coastlines)
        col += 1
        for src in sources:  # drift panels
            dr = drs[src.run]
            gm = float(dr.weighted(np.cos(np.deg2rad(dr.latitude)))
                       .mean(["longitude", "latitude"]))
            P.map_panel(axes[col], dr, cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                        title=f"{src.pretty} drift = fc − {ref_src.ref_label}\n"
                              f"({area} mean {gm:+.4g} {units})",
                        cbar_label=f"drift [{units}]", extent=extent, fig=fig,
                        coast=P.draw_coastlines)
            col += 1

        models = " vs ".join(dict.fromkeys(x.pretty for x in sources))
        fig.suptitle(f"{ref_src.ref_label} — mean 10-day {label}@{lev} hPa drift, "
                     f"{area} ({models}, Rackow et al. 2024 Fig. 3 style)",
                     y=1.04, fontsize=13)
        fig.tight_layout()
        out = figdir / fig_naming.figure_name(
            S.model_token(sources), ref_src.dataset, reg, var, lev, ref_src.year,
            fig_naming.months_token(period), "drift_maps")
        P.save_fig(fig, out)


def plot_variable(sources, var, levels, regions, periods):
    meta = C.VARIABLES[var]
    short, units, label, fcmap = (meta["short"], meta["units"],
                                  meta["label"], meta["cmap"])
    print(f"=== drift maps: {var} ({short}) ===")
    for period in periods:
        fields = {}
        for s in sources:
            out = s.day10_fields(var, short, levels, period)
            if out is not None:
                fields[s.run] = out
        if not fields:
            continue
        # keep only sources that actually produced fields, preserving order
        srcs = [s for s in sources if s.run in fields]
        for reg in regions:
            plot_period_region(fields, srcs, var, short, units, label, fcmap,
                               C.render_levels(levels), period, reg)
        for out in fields.values():
            out.close()
        print(f"  saved {C.period_dir_name(period)} x {len(regions)} region(s)")


def main():
    sources = S.resolve_sources()
    levels = S.requested_levels(sources)
    regions = C.selected_regions()
    periods = C.selected_periods()
    for var in C.selected_variables():
        plot_variable(sources, var, levels, regions, periods)
    print(f"done -> {C.FIG_ROOT}/{S.run_label(sources)}/<period>/<region>/<variable>/drift_maps/")


if __name__ == "__main__":
    main()
