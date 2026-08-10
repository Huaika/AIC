#!/usr/bin/env python
"""Forecast-error map grids (rows = lead time, columns = model).

Shared renderer for the OOD and case-study "regional plots": a grid of mean
forecast-error fields with rows = lead {+1, +5, +10 days} and columns = models,
one per year, plus one combined figure across years (year super-headers spanning
each year's model columns). One shared diverging 'Error [units]' colour scale
across every figure, no titles, no lat/lon axis labels.

The data (``fields[year][model][lead] -> 2-D error DataArray``) is gathered by the
caller (view.ood over the world; controller.casestudy.plots over the heatwave
footprint), so this module only lays the panels out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aic.controller.eval import sources as S
from aic.view import plotting as P

LEADS = [24, 120, 240]                                  # hours
LEAD_LABELS = {24: "+1 day", 120: "+5 days", 240: "+10 days"}
MODEL_ORDER = ["graphcast", "neuralgcm"]                # GraphCast column first


def _order(models):
    return ([m for m in MODEL_ORDER if m in models]
            + [m for m in models if m not in MODEL_ORDER])


def render_error_maps(fields, years, units, *, extent, coast, out_dir, stem, fmts=None):
    """Write a per-year error-map grid for each year + one combined grid across years.

    ``fields[year][model][lead]`` is a 2-D error DataArray (already masked to the
    domain) or None. All figures share one symmetric colour scale (99th-pct of |error|
    over every panel), so years/leads/models are directly comparable."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cbar = f"Error [{units}]"
    row_labels = [LEAD_LABELS[L] for L in LEADS]
    allc = [fields[y][m].get(L) for y in years for m in fields[y] for L in LEADS]
    allc = [c for c in allc if c is not None]
    vlim = max((float(np.nanpercentile(np.abs(c.values), 99)) for c in allc),
               default=1.0) or 1.0

    for y in years:                                     # one figure per year
        models = _order(list(fields[y]))
        cells = [[fields[y][m].get(L) for m in models] for L in LEADS]
        fig = P.error_map_grid(
            cells, row_labels=row_labels,
            col_labels=[S.MODEL_PRETTY.get(m, m) for m in models],
            extent=extent, cbar_label=cbar, coast=coast, vlim=vlim)
        for p in P.save_fig(fig, out_dir / f"{stem}_{y}.pdf", fmts=fmts):
            print(f"[errmap] wrote {p.name}", flush=True)

    cols, group, ci = [], [], 0                          # combined: year x model
    for y in years:
        models = _order(list(fields[y]))
        group.append((y, ci, ci + len(models) - 1))
        cols += [(y, m) for m in models]
        ci += len(models)
    cells = [[fields[y][m].get(L) for (y, m) in cols] for L in LEADS]
    fig = P.error_map_grid(
        cells, row_labels=row_labels,
        col_labels=[S.MODEL_PRETTY.get(m, m) for (_, m) in cols],
        group_labels=group, extent=extent, cbar_label=cbar, coast=coast, vlim=vlim)
    yrs = "-".join(str(y) for y in years)
    for p in P.save_fig(fig, out_dir / f"{stem}_{yrs}_combined.pdf", fmts=fmts):
        print(f"[errmap] wrote {p.name}", flush=True)
