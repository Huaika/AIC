#!/usr/bin/env python
"""Shared matplotlib primitives for the diagnostic views.

Every diagnostic figure family -- spaghetti (``view/spaghetti``), drift skill
(``view/drift``), drift maps (``view/drift_maps``), the out-of-distribution
analysis (``view/ood``) and the heat-wave case study
(``controller/casestudy/plots``) -- shares a handful of drawing idioms: the thin
per-init rollout "bundle", a map panel, a skill (RMSE/bias) curve, the day-of-year
x-axis for overlaying different years, and month ticks. They live here ONCE so the
plotters stay thin and consistent (model colours come from ``sources.MODEL_COLORS``).
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

INK = "#222222"
GRID = "#dddddd"


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
    ``metric`` column (as produced by ``view.drift.aggregate``)."""
    if zero_line:
        ax.axhline(0.0, color="0.4", lw=0.8, ls=":", alpha=0.6)
    for color, label, df in curves:
        ax.plot(df["lead_day"], df[metric], color=color, lw=lw, label=label)
    ax.set_xlabel("lead time (days)")
    ax.grid(True, alpha=0.3)


def wrap_lon180(field):
    """Put a field's longitude on the -180..180 convention (sorted ascending), so it
    aligns with ``region_extent`` boxes. Model-grid fields are stored on 0..360, so
    without this the western hemisphere (real -180..0, stored 180..360) would be drawn
    off the right edge of a -180..180 extent and silently vanish."""
    lon = ((field.longitude + 180) % 360) - 180
    return field.assign_coords(longitude=lon).sortby("longitude")


def map_panel(ax, field, *, cmap, vmin, vmax, title, cbar_label, extent, fig,
              coast=None):
    """One pcolormesh map panel with a colourbar, optional coastlines and a
    ``(lon_w, lon_e, lat_s, lat_n)`` extent."""
    field = wrap_lon180(field)
    m = ax.pcolormesh(field.longitude, field.latitude, field, cmap=cmap,
                      vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(m, ax=ax, shrink=0.82, label=cbar_label)
    if coast is not None:
        coast(ax)
    w, e, s, n = extent
    ax.set_xlim(w, e); ax.set_ylim(s, n)
    ax.set_title(title, fontsize=10); ax.grid(alpha=0.2)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
