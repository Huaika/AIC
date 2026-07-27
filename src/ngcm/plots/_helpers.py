"""Shared helpers for the ``ngcm.plots.*`` scripts.

The four plot scripts (``drift``, ``drift_maps``, ``spaghetti``, ``drift_anim``)
used to duplicate the same boilerplate:

* iterate prediction files with progress logging,
* read a tidy CSV cache and recompute only when a required region is missing,
* drive the standard ``for var in selected_variables: plot_variable(...)`` loop,
* draw the 3-panel *forecast | reference | difference* map,
* save a figure with the canonical ``fig_naming`` and close it.

These helpers keep behavior identical while removing the duplication.
"""
from __future__ import annotations

from typing import Callable, Iterable, Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ngcm import eval_common as C
from common import fig_naming


# --------------------------------------------------------------------------- #
# Prediction files
# --------------------------------------------------------------------------- #
def prediction_files() -> list:
    """All ``pred_<YEAR>_*.nc`` rollouts for the active run, sorted."""
    return sorted(C.PRED_DIR.glob(f"pred_{C.YEAR}_*.nc"))


def pred_init_date(ds: xr.Dataset, path) -> pd.Timestamp:
    """Init date of a rollout dataset (attr with filename fallback)."""
    return pd.to_datetime(
        ds.attrs.get("init_date", path.stem.replace(f"pred_{C.YEAR}_", ""))
    )


def iter_predictions(files: Sequence, tag: str) -> Iterator[tuple[int, object, xr.Dataset]]:
    """Yield ``(i, path, ds)`` for each file, logging progress every 50 files.

    The caller is responsible for closing ``ds``.
    """
    n = len(files)
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        yield i, f, ds
        if i % 50 == 0 or i == n - 1:
            print(f"  {i + 1}/{n}")
    _ = tag  # kept for future per-tag logging without changing the signature


# --------------------------------------------------------------------------- #
# Cached tidy CSV keyed by (region, ...)
# --------------------------------------------------------------------------- #
def load_region_cache(csv, regions: Iterable[str], tag: str) -> pd.DataFrame | None:
    """Return the cached frame iff it already covers every requested region.

    Prints the same ``[tag] cached`` / ``[tag] cache ... missing regions`` lines
    as the previous inline code so log output is unchanged.
    """
    if not csv.exists():
        return None
    df = pd.read_csv(csv, parse_dates=["init_date"])
    if "region" in df.columns and set(regions) <= set(df["region"].unique()):
        print(f"[{tag}] cached {csv}")
        return df
    print(f"[{tag}] cache {csv} missing regions -> recompute")
    return None


# --------------------------------------------------------------------------- #
# Figure I/O
# --------------------------------------------------------------------------- #
def save_and_close(fig, figdir, region, var, level, period, kind, ext="pdf"):
    """Save ``fig`` under ``figdir`` with the canonical name and close it."""
    out = figdir / fig_naming.figure_name(
        C.MODEL, C.DATASET, region, var, level, C.YEAR,
        fig_naming.months_token(period), kind, ext=ext,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 3-panel forecast | reference | difference map
# --------------------------------------------------------------------------- #
def plot_three_panel_maps(
    axes,
    fig,
    lon,
    lat,
    forecast: xr.DataArray,
    reference: xr.DataArray,
    diff: xr.DataArray,
    *,
    field_cmap: str,
    units: str,
    field_label: str,
    diff_label: str = "difference",
    titles: tuple[str, str, str],
    extent: tuple[float, float, float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    dlim: float | None = None,
):
    """Draw the standard forecast | reference | diff pcolormesh triptych.

    Color scales default to the shared field range (forecast+reference) and the
    99th absolute percentile of the difference (diverging, symmetric).
    """
    if vmin is None or vmax is None:
        vmin = float(min(forecast.min(), reference.min()))
        vmax = float(max(forecast.max(), reference.max()))
    if dlim is None:
        dlim = float(np.nanpercentile(np.abs(np.asarray(diff)), 99)) or 1.0

    panels = (
        (axes[0], forecast, titles[0], field_cmap, vmin, vmax, f"{field_label} [{units}]"),
        (axes[1], reference, titles[1], field_cmap, vmin, vmax, f"{field_label} [{units}]"),
        (axes[2], diff, titles[2], "RdBu_r", -dlim, dlim, f"{diff_label} [{units}]"),
    )
    for ax, fld, ttl, cm, lo, hi, cl in panels:
        m = ax.pcolormesh(lon, lat, fld, cmap=cm, vmin=lo, vmax=hi, shading="auto")
        C.draw_coastlines(ax)
        ax.set_title(ttl)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        if extent is not None:
            w, e, s, n_ = extent
            ax.set_xlim(w, e)
            ax.set_ylim(s, n_)
        ax.grid(alpha=0.2)
        fig.colorbar(m, ax=ax, shrink=0.8, label=cl)
    return vmin, vmax, dlim


# --------------------------------------------------------------------------- #
# Convenience labels
# --------------------------------------------------------------------------- #
def area_name(region: str) -> str:
    """Display name for a region (``world`` -> ``global``)."""
    return "global" if region == "world" else region


# --------------------------------------------------------------------------- #
# Standard main() for the per-variable plot scripts
# --------------------------------------------------------------------------- #
def run_for_variables(plot_variable: Callable, kind_subdir: str) -> None:
    """The ``main`` shared by drift / drift_maps / spaghetti."""
    levels = C.requested_levels()
    regions = C.selected_regions()
    periods = C.selected_periods()
    for var in C.selected_variables():
        plot_variable(var, levels, regions, periods)
    print(f"done -> {C.FIGROOT}/<period>/<region>/<variable>/{kind_subdir}/")
