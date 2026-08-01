#!/usr/bin/env python
"""A framework to detect and CATEGORIZE heat waves from ERA5 850 hPa temperature.

Definition implemented (user-specified):
    A heat wave is a period of AT LEAST 3 CONSECUTIVE DAYS during which the
    00 UTC 850 hPa temperature exceeds the HIGHEST 5 % (i.e. the 95th percentile)
    of the values that fall within a +/-WINDOW-day window around that calendar day,
    over the 1991-2020 reference period.

    The threshold is therefore a smooth day-of-year climatological percentile
    (one value per calendar day, pooled over the reference years and a window of
    neighbouring days). A region "experiences" a heat wave when its representative
    T850 (area-weighted mean by default) stays above that day's threshold for >= 3
    consecutive days -- heat waves are treated per singular region (they may then
    move on or dissipate), so detection runs on a per-region 1-D daily series.

Pipeline (all steps are independent, composable functions):
    1. regional_series()          area-weighted-mean T850 daily series for a region
    2. doy_threshold()            per-calendar-day 95th-pct threshold (+/-window)
    3. exceedance()               value - threshold[doy]  (>0 == a "hot" day)
    4. find_events()              >= 3-consecutive-hot-day spells -> [Heatwave]
    5. build_scheme()/categorize  severity classes from the region's own
                                  reference-period heat-wave magnitude distribution

Each Heatwave carries: start/end/duration, peak T850, peak & mean & cumulative
exceedance (K and K*days), and a severity category. "Cumulative exceedance"
(sum over the spell of T850-threshold, in K*days) is the magnitude used for the
severity categories -- it captures both how hot and how long the event was.

Reference period is configurable but defaults to 1991-2020 per the definition.
Window default is +/-5 days ("a window of 10 days around the day"); override with
HW_WINDOW / the `window` argument if you read it as +/-10.

The module is dependency-light (numpy/pandas/xarray) and self-contained (its
REGIONS mirror aic.controller.eval.eval_common.REGIONS). Import the functions, or
run it as a driver (see __main__) to build a catalog for one or more regions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Named regions: single source of truth (a region may also be given as a single
# point via point=(lat, lon) to regional_series()).
from aic.regions import REGIONS, select_region

# Default severity classes, keyed by where an event's cumulative-exceedance
# magnitude falls in the reference-period distribution of heat-wave magnitudes for
# THAT region (so the classes are region-relative, per the definition's framing).
# (lower-quantile inclusive, upper-quantile exclusive) -> label.
DEFAULT_SEVERITY_BINS = [
    (0.00, 0.50, "moderate"),
    (0.50, 0.80, "strong"),
    (0.80, 0.95, "severe"),
    (0.95, 1.01, "extreme"),
]


# --------------------------------------------------------------------------- #
# 1. regional daily T850 series
# --------------------------------------------------------------------------- #
def regional_series(files, region="europe", reduce="mean",
                    point=None, var="temperature") -> xr.DataArray:
    """Collapse the gridded daily T850 to a single daily series for a region.

    files : list of the per-year NetCDFs (t850_24h_*.nc), opened lazily.
    reduce: 'mean' (area-weighted, default) | 'median' | 'max' over the region.
    point : (lat, lon) to take the nearest grid cell instead of a box.
    Returns a 1-D DataArray indexed by daily 'time' (K)."""
    ds = xr.open_mfdataset(sorted(files), combine="by_coords")
    da = ds[var]
    if point is not None:
        lat, lon = point
        lon = ((lon + 180) % 360) - 180
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180)).sortby("longitude")
        series = da.sel(latitude=lat, longitude=lon, method="nearest")
    else:
        da = select_region(da, region)
        if reduce == "mean":
            w = np.cos(np.deg2rad(da.latitude))
            series = da.weighted(w).mean(["latitude", "longitude"])
        elif reduce == "median":
            series = da.median(["latitude", "longitude"])
        elif reduce == "max":
            series = da.max(["latitude", "longitude"])
        else:
            raise ValueError(f"reduce must be mean|median|max (got {reduce!r})")
    return series.load()


# --------------------------------------------------------------------------- #
# 2. day-of-year percentile threshold (the "highest 5 %" over +/-window days)
# --------------------------------------------------------------------------- #
def doy_threshold(series: xr.DataArray, q: float = 0.95, window: int = 5,
                  ref_years=(1991, 2020), n_doy: int = 366) -> np.ndarray:
    """Per-calendar-day q-quantile of `series` over the reference years, pooling a
    circular +/-window of neighbouring calendar days.

    Returns thr[1..n_doy] (thr[0] unused / NaN). Vectorised over the 366 calendar
    days; the window wraps across the year boundary so late-Dec and early-Jan share
    neighbours. Feb-29 (doy 60 in leap years) is included where present."""
    t = pd.to_datetime(series["time"].values)
    lo, hi = ref_years
    mask = (t.year >= lo) & (t.year <= hi)
    vals = np.asarray(series.values, float)[mask]
    doy = t.dayofyear.values[mask]
    thr = np.full(n_doy + 1, np.nan)
    for d in range(1, n_doy + 1):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, n_doy - dist)   # circular distance
        sel = vals[dist <= window]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            thr[d] = np.quantile(sel, q)
    return thr


def exceedance(series: xr.DataArray, thr: np.ndarray):
    """Map each day to its day-of-year threshold and return (dates, values,
    anomaly=value-threshold, hot=anomaly>0)."""
    t = pd.to_datetime(series["time"].values)
    vals = np.asarray(series.values, float)
    day_thr = thr[t.dayofyear.values]
    anom = vals - day_thr
    hot = np.isfinite(anom) & (anom > 0)
    return t, vals, anom, hot


# --------------------------------------------------------------------------- #
# 3. spell detection + per-event metrics
# --------------------------------------------------------------------------- #
@dataclass
class Heatwave:
    region: str
    start: str            # ISO date of first hot day
    end: str              # ISO date of last hot day
    duration_days: int
    peak_date: str        # date of the hottest day (max T850)
    peak_t850_K: float
    peak_exceedance_K: float      # max (T850 - threshold) over the spell
    mean_exceedance_K: float
    cumulative_exceedance_Kdays: float   # magnitude = sum of exceedances
    category: str = "unclassified"
    severity_rank: float = float("nan")  # 0..1 position in the ref distribution


def _runs(hot: np.ndarray, min_len: int):
    """Yield (i0, i1) half-open index ranges of consecutive True with length>=min_len."""
    i = 0
    n = len(hot)
    while i < n:
        if hot[i]:
            j = i
            while j < n and hot[j]:
                j += 1
            if j - i >= min_len:
                yield i, j
            i = j
        else:
            i += 1


def find_events(region, dates, vals, anom, hot, min_duration=3) -> list[Heatwave]:
    """Detect >= min_duration consecutive hot days -> Heatwave events with metrics."""
    events = []
    for i0, i1 in _runs(hot, min_duration):
        d = dates[i0:i1]
        v = vals[i0:i1]
        a = anom[i0:i1]
        kpeak = int(np.argmax(v))
        events.append(Heatwave(
            region=region,
            start=str(d[0].date()),
            end=str(d[-1].date()),
            duration_days=int(i1 - i0),
            peak_date=str(d[kpeak].date()),
            peak_t850_K=float(v[kpeak]),
            peak_exceedance_K=float(np.max(a)),
            mean_exceedance_K=float(np.mean(a)),
            cumulative_exceedance_Kdays=float(np.sum(a)),
        ))
    return events


# --------------------------------------------------------------------------- #
# 4. severity categorisation (region-relative)
# --------------------------------------------------------------------------- #
def build_scheme(ref_events: list[Heatwave], bins=DEFAULT_SEVERITY_BINS):
    """Turn the reference-period events' magnitudes into quantile cut points, so a
    category is 'where this event's magnitude sits among that region's historical
    heat waves'. Returns (sorted_magnitudes, bins)."""
    mags = np.sort([e.cumulative_exceedance_Kdays for e in ref_events])
    return mags, bins


def categorize(events: list[Heatwave], scheme) -> list[Heatwave]:
    """Assign each event a severity_rank (empirical CDF position of its magnitude in
    the reference distribution) and a category label from the bins."""
    mags, bins = scheme
    n = len(mags)
    for e in events:
        if n == 0:
            e.severity_rank = float("nan"); e.category = "unclassified"; continue
        # fraction of reference events with magnitude <= this one
        rank = float(np.searchsorted(mags, e.cumulative_exceedance_Kdays,
                                     side="right")) / n
        e.severity_rank = rank
        for lo, hi, label in bins:
            if lo <= rank < hi:
                e.category = label
                break
    return events


def catalog(files, region="europe", q=0.95, window=5, min_duration=3,
            ref_years=(1991, 2020), reduce="mean", point=None,
            bins=DEFAULT_SEVERITY_BINS):
    """End-to-end: build the region's daily series, threshold, detect + categorise
    heat waves over the WHOLE series (calibrating categories on the events found in
    the reference sub-period). Returns (events, threshold, series)."""
    series = regional_series(files, region=region, reduce=reduce, point=point)
    thr = doy_threshold(series, q=q, window=window, ref_years=ref_years)
    dates, vals, anom, hot = exceedance(series, thr)
    events = find_events(region, dates, vals, anom, hot, min_duration=min_duration)
    # calibrate severity on events whose PEAK falls inside the reference period
    lo, hi = ref_years
    ref_events = [e for e in events if lo <= int(e.peak_date[:4]) <= hi]
    scheme = build_scheme(ref_events, bins=bins)
    events = categorize(events, scheme)
    return events, thr, series


def to_dataframe(events: list[Heatwave]) -> pd.DataFrame:
    return pd.DataFrame([asdict(e) for e in events])


# --------------------------------------------------------------------------- #
# driver -- build a catalog for one or more regions and save CSVs
# --------------------------------------------------------------------------- #
def _default_files():
    d = Path(os.environ.get(
        "HW_DATA_DIR",
        "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/era5_heatwave"))
    return sorted(d.glob("t850_24h_world_*.nc"))


def main():
    files = _default_files()
    if not files:
        raise SystemExit("no t850_24h_world_*.nc found (set HW_DATA_DIR)")
    regions = os.environ.get("HW_REGIONS", "europe").replace(",", " ").split()
    # optional point analyses: HW_POINTS="lat,lon; lat,lon" (nearest grid cell) --
    # the intended per-location scale for detecting localized heat waves.
    points = []
    for pair in os.environ.get("HW_POINTS", "").split(";"):
        pair = pair.strip()
        if pair:
            la, lo = (float(x) for x in pair.split(","))
            points.append((la, lo))
    q = float(os.environ.get("HW_Q", "0.95"))
    window = int(os.environ.get("HW_WINDOW", "5"))
    min_dur = int(os.environ.get("HW_MIN_DURATION", "3"))
    reduce = os.environ.get("HW_REDUCE", "mean")
    outdir = Path(os.environ.get(
        "HW_CATALOG_DIR",
        "/pfs/work9/workspace/scratch/ka_dm9435-ai-climate/heatwave_catalog"))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[hw] {len(files)} yearly files; regions={regions} q={q} "
          f"window=+/-{window}d min_duration={min_dur} reduce={reduce}", flush=True)

    for region in regions:
        events, thr, series = catalog(files, region=region, q=q, window=window,
                                      min_duration=min_dur, reduce=reduce)
        df = to_dataframe(events)
        csv = outdir / f"heatwaves_{region}_q{int(q*100)}_w{window}.csv"
        df.to_csv(csv, index=False)
        # persist the day-of-year threshold too
        thr_da = xr.DataArray(thr[1:], dims=["dayofyear"],
                              coords={"dayofyear": np.arange(1, len(thr))},
                              name="t850_threshold_K")
        thr_da.to_netcdf(outdir / f"threshold_{region}_q{int(q*100)}_w{window}.nc")
        n = len(events)
        by_cat = df["category"].value_counts().to_dict() if n else {}
        longest = int(df["duration_days"].max()) if n else 0
        print(f"[hw] {region}: {n} heat waves ({series.sizes['time']} days), "
              f"longest {longest} d, by category {by_cat} -> {csv.name}", flush=True)

    for (la, lo) in points:
        label = f"pt_{la:g}_{lo:g}"
        events, thr, series = catalog(files, region=label, q=q, window=window,
                                      min_duration=min_dur, point=(la, lo))
        df = to_dataframe(events)
        csv = outdir / f"heatwaves_{label}_q{int(q*100)}_w{window}.csv"
        df.to_csv(csv, index=False)
        top = (df.sort_values("cumulative_exceedance_Kdays", ascending=False)
                 .head(5)) if len(df) else df
        print(f"[hw] {label}: {len(df)} heat waves; top-5 by magnitude:", flush=True)
        for _, r in top.iterrows():
            print(f"    {r['start']}..{r['end']} ({r['duration_days']:>2}d) "
                  f"peakT={r['peak_t850_K']:.1f}K cumExc={r['cumulative_exceedance_Kdays']:.1f} "
                  f"[{r['category']}]", flush=True)


if __name__ == "__main__":
    main()
