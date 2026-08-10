#!/usr/bin/env python
"""Shared matplotlib primitives for the diagnostic views.

Every diagnostic figure family -- spaghetti (``view/spaghetti``), drift skill
(``view/drift``), drift maps (``view/drift_maps``), the out-of-distribution
analysis (``view/ood``) and the heatwave case study
(``controller/casestudy/plots``) -- shares a handful of drawing idioms: the thin
per-init rollout "bundle", a map panel, a skill (RMSE/bias) curve, the day-of-year
x-axis for overlaying different years, and month ticks. They live here ONCE so the
plotters stay thin and consistent (model colours come from ``sources.MODEL_COLORS``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from aic import config
from aic.config import COAST_ZARR
from aic.regions import to_lon180
from aic.style import INK, GRID, MUTED  # re-exported for the view modules' back-compat

# default output file type for every figure; override per call (save_fig(fmts=...))
# or globally via AIC_FIG_FMT.
DEFAULT_FIG_FMT = config.env_str("AIC_FIG_FMT", "pdf")


def save_fig(fig, path, *, fmts=None, dpi=150, close=True):
    """Write ``fig`` to ``path`` and close it (unless ``close=False``). The ONE place
    figures are saved, so the file type is chosen consistently: format precedence is
    ``fmts`` (a list, to emit several types at once) > ``path``'s own suffix > the
    package default (``DEFAULT_FIG_FMT`` / ``AIC_FIG_FMT``). ``bbox_inches='tight'``
    always; ``dpi`` applies to raster formats (ignored for pdf). Returns the written
    paths."""
    path = Path(path)
    if not fmts:
        fmts = [path.suffix[1:]] if path.suffix else [DEFAULT_FIG_FMT]
    written = []
    for fmt in fmts:
        p = path.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    if close:
        plt.close(fig)
    return written

# land-sea mask backdrop (grid-independent; the same NextGEMS constant-fields mask
# is reused for every run and cached on first use)
_COAST_ZARR = COAST_ZARR
_COAST = None


def doy_axis(dates):
    """Map datetimes to a common leap reference year (2000) so series from
    different calendar years overlay on one Jan-Dec axis. Accepts a Series, list,
    array or index (a plain ``.dayofyear`` only exists on a DatetimeIndex, not on a
    datetime Series -- hence the explicit DatetimeIndex)."""
    doy = pd.DatetimeIndex(pd.to_datetime(dates)).dayofyear
    return pd.Timestamp("2000-01-01") + pd.to_timedelta(doy - 1, unit="D")


def month_axis(ax):
    """Label the x-axis by month."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())


def despine(ax, sides=("top", "right")):
    for s in sides:
        ax.spines[s].set_visible(False)


def draw_rollout_bundle(ax, roll, color, *, linestyle="-", lw=0.6, alpha=0.5,
                        x_transform=None, every=1, zorder=2):
    """Draw one forecast source's thin rollout lines: one per init-day, each
    collapsed to per-lead-day means. ``roll`` needs columns ``init_date``,
    ``lead_hours`` and ``pred_gmean``. ``x_transform`` maps the valid-time series
    onto the x-axis (e.g. :func:`doy_axis` for multi-year overlays); the default is
    identity (absolute valid time)."""
    r = roll.copy()
    r["valid_time"] = r["init_date"] + pd.to_timedelta(r["lead_hours"], unit="h")
    r["lead_day_idx"] = (r["lead_hours"] // 24).astype(int)
    xt = x_transform or (lambda s: s)
    for d in sorted(r["init_date"].unique())[::every]:
        g = r[r["init_date"] == d]
        daily = (g.groupby("lead_day_idx")
                   .agg(vt=("valid_time", "mean"), val=("pred_gmean", "mean"))
                   .reset_index())
        ax.plot(xt(daily["vt"]), daily["val"], color=color, lw=lw, ls=linestyle,
                alpha=alpha, zorder=zorder)


def draw_skill_metric(ax, curves, metric, *, zero_line=False, lw=1.9):
    """Draw one skill metric (``"rmse"`` or ``"bias"``) vs lead day. ``curves`` is a
    list of ``(color, label, dataframe)`` where each frame has ``lead_day`` and the
    ``metric`` column (as produced by ``eval_common.aggregate``). If the frame also
    carries ``<metric>_lo``/``<metric>_hi`` (bootstrap CI), a matching shaded band is
    drawn."""
    if zero_line:
        ax.axhline(0.0, color="0.4", lw=0.8, ls=":", alpha=0.6)
    banded = False
    for color, label, df in curves:
        lo, hi = f"{metric}_lo", f"{metric}_hi"
        if lo in df.columns and hi in df.columns:
            ax.fill_between(df["lead_day"], df[lo], df[hi], color=color,
                            alpha=0.18, lw=0)
            banded = True
        ax.plot(df["lead_day"], df[metric], color=color, lw=lw, label=label)
    if banded:
        pct = int(round(config.env_float("AIC_BOOT_CI", 0.95) * 100))
        ax.plot([], [], color="0.5", lw=6, alpha=0.25, label=f"{pct}% CI")
    ax.set_xlabel("Lead Time [days]")
    ax.grid(True, alpha=0.3)


def skill_ylabel(metric, var_label, level, units):
    """Shared y-axis label for a skill plot, e.g. 'Weighted Temperature Mean Bias at
    850 hPa [K]' (bias) / 'Weighted Geopotential RMSE at 500 hPa [m^2/s^2]' (rmse).
    'Weighted' because the spatial mean is cos(lat) area-weighted."""
    stat = "Mean Bias" if metric == "bias" else "RMSE"
    return f"Weighted {str(var_label).capitalize()} {stat} at {level} hPa [{units}]"


def spaghetti_ylabel(var_label, level, units, area="global"):
    """Y-axis label for a spaghetti plot, e.g. 'Global Mean Temperature at 850 hPa [K]'
    (area='global' -> 'Global'; otherwise the region name)."""
    return f"{str(area).capitalize()} Mean {str(var_label).capitalize()} at {level} hPa [{units}]"


def skill_facets(panels, metric, *, ylabel, zero_line=False, ylim=None,
                 per_panel=(5.0, 4.4)):
    """One figure with a column per (title, curves) panel, sharing the y-axis so the
    panels are directly comparable: each panel titled (e.g. the year) with its own
    'Lead Time [days]' x-label, ONE shared y-label on the left, ONE legend. ``curves``
    is ``[(color, label, df)]`` as for draw_skill_metric (bands drawn if present).
    ``ylim`` fixes the shared y-range (else autoscaled)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(per_panel[0] * n, per_panel[1]),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (title, curves) in zip(axes, panels):
        draw_skill_metric(ax, curves, metric, zero_line=zero_line)
        ax.set_title(str(title), fontsize=12)
        despine(ax)
        # each year-panel keeps its own legend, auto-placed so it clears that
        # panel's data (the panels can have quite different data locations)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
    if ylim is not None:
        axes[0].set_ylim(*ylim)     # sharey -> applies to all panels
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def map_scales(field_arrays, drift_arrays, drift_pct=99):
    """Shared colour limits for a drift-map row: a common ``(vmin, vmax)`` over all
    field panels (forecasts + reference) and a symmetric ``±dlim`` for the drift /
    error panels (``drift_pct``-th percentile of ``|drift|``). NaN-aware, so it works
    for both full-region crops and NaN-masked footprints."""
    vmin = float(np.nanmin([np.nanmin(f.values) for f in field_arrays]))
    vmax = float(np.nanmax([np.nanmax(f.values) for f in field_arrays]))
    dlim = max((float(np.nanpercentile(np.abs(d.values), drift_pct))
                for d in drift_arrays), default=1.0) or 1.0
    return vmin, vmax, dlim


def map_panel(ax, field, *, cmap, vmin, vmax, title, cbar_label, extent, fig,
              coast=None):
    """One pcolormesh map panel with a colourbar, optional coastlines and a
    ``(lon_w, lon_e, lat_s, lat_n)`` extent. The field is put on the -180..180
    longitude convention first so western-hemisphere cells align with the extent."""
    field = to_lon180(field)
    m = ax.pcolormesh(field.longitude, field.latitude, field, cmap=cmap,
                      vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(m, ax=ax, shrink=0.82, label=cbar_label)
    if coast is not None:
        coast(ax)
    w, e, s, n = extent
    ax.set_xlim(w, e); ax.set_ylim(s, n)
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.2)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")


def draw_coastlines(ax, lw=0.4, color="k", alpha=0.7):
    """Coastlines by contouring the 0.25deg land-sea mask at 0.5 (no cartopy). A
    grid-independent backdrop overlay, cached after the first call. Single source
    for the coastline overlay used by the drift maps, the case study and the GIFs."""
    global _COAST
    if _COAST is None:
        lsm = xr.open_zarr(_COAST_ZARR)["land_sea_mask"]
        lsm = (lsm.assign_coords(lon=(((lsm.lon + 180) % 360) - 180))
               .sortby("lon").sortby("lat"))
        _COAST = (lsm.lon.values, lsm.lat.values, lsm.values)
    lon, lat, mask = _COAST
    ax.contour(lon, lat, mask, levels=[0.5], colors=color, linewidths=lw, alpha=alpha)


def error_map_grid(cells, *, row_labels, col_labels, extent, cbar_label,
                   group_labels=None, coast=None, cmap="RdBu_r", vlim=None,
                   drift_pct=99):
    """Grid of forecast-error maps sharing ONE diverging colour scale.

    ``cells[r][c]`` is a 2-D error DataArray (or None). Columns are models
    (``col_labels``, drawn as headers on top), rows are lead times (``row_labels`` like
    '+1 day', drawn on the left). ``group_labels`` = list of ``(text, c0, c1)`` draws a
    second header row spanning columns c0..c1 inclusive (e.g. the year over its two
    model columns). No titles, no lat/lon axis labels; one shared 'Error [units]'
    colourbar (symmetric +/- vlim) on the right. Returns the figure."""
    nr, nc = len(row_labels), len(col_labels)
    if vlim is None:
        vals = [c for row in cells for c in row if c is not None]
        vlim = max((float(np.nanpercentile(np.abs(c.values), drift_pct)) for c in vals),
                   default=1.0) or 1.0
    fig, axes = plt.subplots(nr, nc, figsize=(2.5 * nc + 1.2, 2.5 * nr + 0.6),
                             squeeze=False, sharex=True, sharey=True)
    w, e, s, n = extent
    m = None
    for r in range(nr):
        for c in range(nc):
            ax = axes[r][c]
            cell = cells[r][c]
            if cell is not None:
                f2 = to_lon180(cell)
                m = ax.pcolormesh(f2.longitude, f2.latitude, f2, cmap=cmap,
                                  vmin=-vlim, vmax=vlim, shading="auto")
                if coast is not None:
                    coast(ax)
            ax.set_xlim(w, e); ax.set_ylim(s, n)
            ax.tick_params(labelbottom=False, labelleft=False, length=0)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=11)
    if m is not None:
        fig.colorbar(m, ax=axes.ravel().tolist(), location="right", shrink=0.9,
                     pad=0.02, label=cbar_label)
    fig.canvas.draw()                         # realise positions for the group headers
    for text, c0, c1 in (group_labels or []):
        x = (axes[0][c0].get_position().x0 + axes[0][c1].get_position().x1) / 2
        top = max(axes[0][c].get_position().y1 for c in range(nc))
        fig.text(x, top + 0.04, str(text), ha="center", va="bottom",
                 fontsize=13, fontweight="bold")
    return fig
