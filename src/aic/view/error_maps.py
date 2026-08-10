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


def render_error_maps(fields, years, units, *, var_label="", extent, coast, out_dir,
                      stem, fmts=None, label=None, cmap="RdBu_r"):
    """Write a per-year error-map grid for each year + one combined grid across years.

    ``fields[year][model][lead]`` is a 2-D error DataArray (already masked to the
    domain) or None. All figures share one symmetric colour scale (99th-pct of |error|
    over every panel), so years/leads/models are directly comparable. ``cmap`` is the
    diverging colormap (per variable; e.g. PuOr_r for geopotential). ``label`` (e.g.
    'heatwave') tags the variant: it is appended to every filename and drawn small in
    the corner of each grid. Leave it None for the default (unlabelled) variant -- the
    OOD grids and the case-study 'yearly' grids -- which keep their clean filenames."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cbar = f"Mean {var_label.title()} Error [{units}]" if var_label else f"Error [{units}]"
    tag = f"_{label}" if label else ""                   # filename suffix for the variant
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
            extent=extent, cbar_label=cbar, coast=coast, vlim=vlim, cmap=cmap,
            spec_label=label)
        for p in P.save_fig(fig, out_dir / f"{stem}{tag}_{y}.pdf", fmts=fmts):
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
        group_labels=group, extent=extent, cbar_label=cbar, coast=coast, vlim=vlim,
        cmap=cmap, spec_label=label)
    yrs = "-".join(str(y) for y in years)
    for p in P.save_fig(fig, out_dir / f"{stem}{tag}_{yrs}_combined.pdf", fmts=fmts):
        print(f"[errmap] wrote {p.name}", flush=True)


def render_error_maps_scoped(fields, years, units, *, var_label="", scopes, extent, coast,
                             out_dir, stem, fmts=None, cmap="RdBu_r"):
    """Error-map grid with the two scopes (whole year / heatwave) side by side.

    ``fields[year][model][scope_key][lead]`` is a 2-D error DataArray (or None). Columns
    nest year > model > scope, so each model shows its 'complete year' panel directly
    beside its 'heatwave' panel; rows are the lead times. ``scopes`` is an ordered list
    of ``(scope_key, column_title)``. One per-year figure (model over scope headers) and
    one combined figure across years (year over model over scope headers), all sharing a
    single symmetric colour scale."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cbar = f"Mean {var_label.title()} Error [{units}]" if var_label else f"Error [{units}]"
    row_labels = [LEAD_LABELS[L] for L in LEADS]
    scope_keys = [k for k, _ in scopes]
    scope_title = dict(scopes)
    allc = [fields[y][m][sk].get(L)
            for y in years for m in fields[y] for sk in fields[y][m] for L in LEADS]
    allc = [c for c in allc if c is not None]
    vlim = max((float(np.nanpercentile(np.abs(c.values), 99)) for c in allc),
               default=1.0) or 1.0

    def _layout(pairs):
        """(cells, col_titles, model_groups, cols) for an ordered list of (year, model)."""
        cols, model_groups, ci = [], [], 0
        for y, m in pairs:
            sks = [sk for sk in scope_keys if sk in fields[y][m]]
            if not sks:
                continue
            model_groups.append((S.MODEL_PRETTY.get(m, m), ci, ci + len(sks) - 1))
            cols += [(y, m, sk) for sk in sks]
            ci += len(sks)
        cells = [[fields[y][m][sk].get(L) for (y, m, sk) in cols] for L in LEADS]
        col_titles = [scope_title[sk] for (_, _, sk) in cols]
        return cells, col_titles, model_groups, cols

    for y in years:                                     # per-year: model > scope
        pairs = [(y, m) for m in _order(list(fields[y]))]
        cells, col_titles, model_groups, cols = _layout(pairs)
        if not cols:
            continue
        fig = P.error_map_grid(
            cells, row_labels=row_labels, col_labels=col_titles, extent=extent,
            cbar_label=cbar, coast=coast, vlim=vlim, cmap=cmap,
            header_levels=[model_groups])
        for p in P.save_fig(fig, out_dir / f"{stem}_{y}.pdf", fmts=fmts):
            print(f"[errmap] wrote {p.name}", flush=True)

    pairs = [(y, m) for y in years for m in _order(list(fields[y]))]  # combined
    cells, col_titles, model_groups, cols = _layout(pairs)
    year_groups = []                                    # year spans its model x scope cols
    for y in years:
        span = [i for i, (yy, _, _) in enumerate(cols) if yy == y]
        if span:
            year_groups.append((y, span[0], span[-1]))
    if cols:
        fig = P.error_map_grid(
            cells, row_labels=row_labels, col_labels=col_titles, extent=extent,
            cbar_label=cbar, coast=coast, vlim=vlim, cmap=cmap,
            header_levels=[model_groups, year_groups])
        yrs = "-".join(str(y) for y in years)
        for p in P.save_fig(fig, out_dir / f"{stem}_{yrs}_combined.pdf", fmts=fmts):
            print(f"[errmap] wrote {p.name}", flush=True)
