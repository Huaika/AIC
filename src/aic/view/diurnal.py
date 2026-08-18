#!/usr/bin/env python
"""Diurnal cycle OF THE DATASETS themselves (no model rollouts involved).

The reanalysis / simulation data are 6-hourly, so every day carries four steps
(00, 06, 12, 18 UTC). This view aggregates them over a whole year:

  value    -- the area-weighted mean field at each UTC hour of day (the mean
              temperature / geopotential itself).
  anomaly  -- the same, minus THAT DAY's mean over its four steps: the offset of
              each 6-hourly step from the daily mean, i.e. the diurnal
              oscillation ("bias" of a synoptic time against the daily mean).

One figure per (variable, level, region): a value panel and an anomaly panel side
by side, one curve per dataset (ERA5 1955 / ERA5 2023 / NextGEMS 2049 ...), so the
size and phase of the daily oscillation are directly comparable across climates.

Input is only the cached model-grid truth (``truth_modelgrid_<short>_<run>.nc``)
that the eval pipeline already built -- the DATASETS, not the forecasts.

    EVAL_RUNS=era5_1955,era5_2023,nextgems2049 \\
    EVAL_VARS=temperature NG_LEVELS=850 EVAL_REGIONS=world,europe \\
      python -m aic.view.diurnal
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aic import config
from aic.controller.eval import eval_common as C
from aic.controller.eval import gridpoints as GP
from aic.controller.eval import sources as S
from aic.view import naming as fig_naming
from aic.view import plotting as P

DIURNAL_ROOT = C.FIG_ROOT / "diurnal_cycle"
DIURNAL_FMTS = config.env_list("DIURNAL_FMTS") or ["pdf", "png"]

HOURS = [0, 6, 12, 18]
CI = config.env_float("AIC_BOOT_CI", 0.95)
_Z = NormalDist().inv_cdf(0.5 + CI / 2.0)     # two-sided normal quantile

# one colour per dataset-era, distinct from the MODEL colours (this figure family
# shows the data, not the forecasts)
ERA_COLORS = {1955: "#4d4d4d", 2023: "#1b7837", 2026: "#7b3294", 2049: "#e08214"}


def _figdir(*parts):
    d = DIURNAL_ROOT.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def era_color(year: int, i: int = 0) -> str:
    return ERA_COLORS.get(int(year), S.PALETTE[i % len(S.PALETTE)])


# --------------------------------------------------------------------------- #
# data reduction: area-mean series -> per-hour composite
# --------------------------------------------------------------------------- #
def series(source, var, lev, region) -> pd.DataFrame:
    """Area-weighted mean of the DATASET field at ``lev`` for every 6-hourly step:
    columns ``time``, ``value``, ``date`` (calendar day), ``hour`` (UTC)."""
    truth = source.truth_at_levels(var, [lev]).sel(level=lev)
    lat, lon = source.prediction_grid()
    pts = GP.GridPoints.from_region(lat, lon, region)
    gm = GP.masked_area_mean(truth, pts).compute()
    t = pd.to_datetime(gm["time"].values)
    return pd.DataFrame({"time": t, "value": np.asarray(gm.values, float),
                         "date": t.normalize(), "hour": t.hour})


def composite(df: pd.DataFrame) -> pd.DataFrame:
    """Per UTC hour of day: the mean value, and the mean ANOMALY against the daily
    mean (each day's four steps minus that day's own mean), with a CI over days.

    Only complete days (all four 6-hourly steps present) enter the anomaly, so the
    daily mean is never taken over a partial day."""
    n_per_day = df.groupby("date")["hour"].nunique()
    full = set(n_per_day[n_per_day == len(HOURS)].index)
    d = df[df["date"].isin(full)].copy()
    d["daily_mean"] = d.groupby("date")["value"].transform("mean")
    d["anom"] = d["value"] - d["daily_mean"]
    g = d.groupby("hour")
    out = pd.DataFrame({
        "hour": g.size().index,
        "value": g["value"].mean(),
        "anom": g["anom"].mean(),
        "n_days": g.size(),
        "anom_sd": g["anom"].std(ddof=1),
        "value_sd": g["value"].std(ddof=1),
    }).reset_index(drop=True)
    for k in ("anom", "value"):
        half = _Z * out[f"{k}_sd"] / np.sqrt(out["n_days"])
        out[f"{k}_lo"], out[f"{k}_hi"] = out[k] - half, out[k] + half
    return out.sort_values("hour")


def _wrap(d: pd.DataFrame) -> pd.DataFrame:
    """Repeat the 00 UTC row at hour 24 so the daily cycle closes visually."""
    if d.empty:
        return d
    first = d.iloc[[0]].copy()
    first["hour"] = 24
    return pd.concat([d, first], ignore_index=True)


# --------------------------------------------------------------------------- #
# figure: rows = variable, columns = year
# --------------------------------------------------------------------------- #
def _hour_axis(ax, bottom=True):
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["00", "06", "12", "18", "24"])
    ax.set_xlim(-0.6, 24.6)
    ax.tick_params(labelbottom=bottom)          # x-tick labels only on the bottom row
    ax.grid(True, alpha=0.3)


def diurnal_grid(cells, specs, years, region, area, year_label):
    """One figure, ``rows = specs`` (variable, level) x ``columns = years``.

    Each cell draws the diurnal cycle as the offset of each synoptic hour from the day's
    own mean (zero-centred, so the oscillation is readable), with a twin RIGHT axis
    carrying the same curve in absolute units. All year-columns in a row share ONE
    zero-centred offset y-axis (``sharey='row'``, same +/-range), so the daily
    oscillations are directly comparable across the years; the twin axis + the daily-mean
    annotation keep each year's absolute level, so no information is lost."""
    nr, nc = len(specs), len(years)
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc + 1.4, 3.7 * nr + 0.5),
                             squeeze=False, sharex=True, sharey="row")
    for r, (var, lev) in enumerate(specs):
        meta = C.VARIABLES[var]
        units, label = meta["units"], meta["label"]
        row = [cells.get((var, y)) for y in years]
        lim = max((float(np.nanmax(np.abs([d["anom_lo"], d["anom_hi"]])))
                   for _, d in filter(None, row)), default=1.0) * 1.15
        for c, y in enumerate(years):
            ax = axes[r][c]
            ax.set_ylim(-lim, lim)                        # same offset axis for every year
            ax.axhline(0.0, color="0.25", lw=1.2, ls=":", alpha=0.85)
            _hour_axis(ax, bottom=(r == nr - 1))
            P.despine(ax, sides=("top",))
            if r == 0:                                   # column title only on the top row
                ax.set_title(year_label.get(y, str(y)))
            item = cells.get((var, y))
            if item is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                        va="center", color=P.MUTED)
                continue
            s, d = item
            dw = _wrap(d)
            col = era_color(s.year, c)
            ax.fill_between(dw["hour"], dw["anom_lo"], dw["anom_hi"], color=col,
                            alpha=0.18, lw=0)
            ax.plot(dw["hour"], dw["anom"], color=col, lw=2.0, marker="o", ms=4.5)
            # twin right axis: the SAME curve read as the absolute mean field
            daily_mean = float(d["value"].mean())
            ax2 = ax.twinx()
            ax2.set_ylim(daily_mean - lim, daily_mean + lim)
            P.despine(ax2, sides=("top",))
            if c == nc - 1:
                ax2.set_ylabel(f"{str(label).capitalize()} at {lev} hPa [{units}]")
            else:
                ax2.tick_params(labelright=False)
            ax.annotate(f"daily mean {daily_mean:.1f} {units}", xy=(0.02, 0.03),
                        xycoords="axes fraction", color=P.MUTED)
            if c == 0:                                   # shared y-label at the row start
                ax.set_ylabel(f"{str(label).capitalize()} @ {lev} hPa\n"
                              f"offset from daily mean [{units}]")
    fig.supxlabel("Time of Day [UTC]")               # one shared x-label for all columns
    fig.tight_layout()
    stem = "-".join(f"{C.VARIABLES[v]['short']}L{l:04d}" for v, l in specs)
    out = _figdir("diurnal_cycle") / (
        f"dataset_{fig_naming.REGION_ABBR.get(region, region)}_{stem}_"
        f"{'-'.join(str(y) for y in years)}_entire-year_diurnal-cycle.pdf")
    for p in P.save_fig(fig, out, fmts=DIURNAL_FMTS):
        print(f"  wrote {p.relative_to(DIURNAL_ROOT)}")


def write_table(cells, specs, years, region):
    """The plotted numbers as one tidy CSV next to the figure."""
    rows = []
    for var, lev in specs:
        for y in years:
            item = cells.get((var, y))
            if item is None:
                continue
            s, d = item
            t = d.copy()
            t.insert(0, "dataset", s.ref_label)
            t.insert(1, "year", s.year)
            t.insert(2, "variable", var)
            t.insert(3, "level", lev)
            t.insert(4, "region", region)
            rows.append(t)
    df = pd.concat(rows, ignore_index=True)
    p = _figdir("diurnal_cycle") / f"diurnal_cycle_{region}.csv"
    df.to_csv(p, index=False)
    print(f"  wrote {p.relative_to(DIURNAL_ROOT)}")


def parse_specs() -> list[tuple[str, int]]:
    """DIURNAL_SPECS='geopotential:500,temperature:850' -> the figure's ROWS."""
    env = config.env_str("DIURNAL_SPECS", "geopotential:500,temperature:850")
    specs = []
    for item in env.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        var, _, lev = item.partition(":")
        if var not in C.VARIABLES:
            raise SystemExit(f"unknown variable {var!r} in DIURNAL_SPECS")
        specs.append((var, int(lev)))
    return specs


def main():
    sources = S.resolve_sources()
    # one entry per DATASET (dataset, year) -- the GraphCast/NeuralGCM runs of the
    # same year share one truth cache, so the second one would plot the same curve.
    seen, datasets = set(), []
    for s in sources:
        if (s.dataset, s.year) not in seen:
            seen.add((s.dataset, s.year))
            datasets.append(s)
    datasets.sort(key=lambda s: s.year)
    years = [s.year for s in datasets]
    year_label = {s.year: s.ref_label for s in datasets}   # "<dataset> <year>" per column
    specs = parse_specs()
    print("[diurnal] datasets: " + ", ".join(s.ref_label for s in datasets))
    print("[diurnal] rows: " + ", ".join(f"{v}@{l}" for v, l in specs))
    for region in C.selected_regions():
        area = "global" if region == "world" else region
        cells = {}
        for var, lev in specs:
            units = C.VARIABLES[var]["units"]
            for s in datasets:
                print(f"=== {var}@{lev} / {region} / {s.ref_label} ===", flush=True)
                c = composite(series(s, var, lev, region))
                print(f"  {int(c['n_days'].iloc[0])} full days, offset range "
                      f"{c['anom'].min():+.3f} .. {c['anom'].max():+.3f} {units}",
                      flush=True)
                cells[(var, s.year)] = (s, c)
        diurnal_grid(cells, specs, years, region, area, year_label)
        write_table(cells, specs, years, region)
    print(f"done -> {DIURNAL_ROOT}/diurnal_cycle/")


if __name__ == "__main__":
    main()
