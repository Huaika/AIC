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


def _panel(ax, field, cmap, vmin, vmax, title, w, e, s, n, cbar_label, fig):
    lon, lat = field.longitude, field.latitude
    m = ax.pcolormesh(lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(title, fontsize=10)
    fig.colorbar(m, ax=ax, shrink=0.8, label=cbar_label)
    C.draw_coastlines(ax)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_xlim(w, e); ax.set_ylim(s, n); ax.grid(alpha=0.2)


def plot_period_region(fields, sources, var, short, units, label, fcmap,
                       levels, period, reg):
    """fields: {run: xr.Dataset(fc_day10_clim, ref_clim, drift)} per source."""
    figdir = S.figure_dir(sources, period, reg, var, "drift_maps")
    w, e, s, n_ = C.region_extent(reg)
    area = "global" if reg == "world" else reg
    ref_src = sources[0]
    for lev in levels:
        fcs = {r: C.select_region(fields[r]["fc_day10_clim"].sel(level=lev), reg)
               for r in fields}
        drs = {r: C.select_region(fields[r]["drift"].sel(level=lev), reg)
               for r in fields}
        rf = C.select_region(fields[ref_src.run]["ref_clim"].sel(level=lev), reg)

        # shared field colour scale over all forecasts + the reference
        allf = list(fcs.values()) + [rf]
        vmin = float(min(f.min() for f in allf))
        vmax = float(max(f.max() for f in allf))
        dlim = max((float(np.nanpercentile(np.abs(d.values), 99)) for d in drs.values()),
                   default=1.0) or 1.0

        npan = 2 * len(sources) + 1
        fig, axes = plt.subplots(1, npan, figsize=(6.3 * npan, 4.3), squeeze=False)
        axes = axes[0]
        col = 0
        for src in sources:  # forecast panels
            ninit = int(fields[src.run].attrs.get("n_inits", 0))
            _panel(axes[col], fcs[src.run], fcmap, vmin, vmax,
                   f"{src.pretty} day-10 climatology\n(mean of {ninit} forecasts)",
                   w, e, s, n_, f"{label} [{units}]", fig)
            col += 1
        _panel(axes[col], rf, fcmap, vmin, vmax,  # shared reference panel
               f"{ref_src.ref_label} reference mean", w, e, s, n_,
               f"{label} [{units}]", fig)
        col += 1
        for src in sources:  # drift panels
            dr = drs[src.run]
            gm = float(dr.weighted(np.cos(np.deg2rad(dr.latitude)))
                       .mean(["longitude", "latitude"]))
            _panel(axes[col], dr, "RdBu_r", -dlim, dlim,
                   f"{src.pretty} drift = fc − {ref_src.ref_label}\n"
                   f"({area} mean {gm:+.4g} {units})",
                   w, e, s, n_, f"drift [{units}]", fig)
            col += 1

        models = " vs ".join(dict.fromkeys(x.pretty for x in sources))
        fig.suptitle(f"{ref_src.ref_label} — mean 10-day {label}@{lev} hPa drift, "
                     f"{area} ({models}, Rackow et al. 2024 Fig. 3 style)",
                     y=1.04, fontsize=13)
        fig.tight_layout()
        out = figdir / fig_naming.figure_name(
            S.model_token(sources), ref_src.dataset, reg, var, lev, ref_src.year,
            fig_naming.months_token(period), "drift_maps")
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


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
