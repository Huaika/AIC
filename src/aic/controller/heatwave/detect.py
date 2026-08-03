#!/usr/bin/env python
"""Definition-driven heat-wave detection on the model grid. ONE code path serves
all three definitions (definitions.py): a day is 'hot' where EVERY daily statistic
the definition names exceeds its day-of-year percentile threshold; a heat wave is a
run of >= MIN_DUR consecutive hot days at a cell."""
from __future__ import annotations

import numpy as np

from aic.controller.heatwave.climatology import doy_percentile, season_percentile

MIN_DUR = 3


def hot_mask(defn, ref_stats, ref_doy, tgt_stats, tgt_doy, window, ref_months=None):
    """Boolean hot mask (T, Y, X) for the target year under `defn`: for each daily
    statistic in defn.stats, the target value exceeds the reference threshold; the
    day is hot where ALL of them do (logical AND).

    Windowed definitions (``defn.kind == "doy"``) compare against a per-calendar-day
    +/-``window`` percentile. The EURO-CORDEX definition (``kind == "season"``)
    compares against a single seasonal per-cell percentile (``ref_months`` gives the
    calendar month of each reference day; ``window`` is ignored)."""
    hot = None
    for s in defn.stats:
        if defn.kind == "season":
            if ref_months is None:
                raise ValueError("seasonal definition needs ref_months")
            thr = season_percentile(ref_stats[s], ref_months, defn.season, defn.pct)
            m = tgt_stats[s] > thr[None, :, :]
        else:
            thr = doy_percentile(ref_stats[s], ref_doy, window, defn.pct)
            m = tgt_stats[s] > thr[tgt_doy - 1]
        hot = m if hot is None else (hot & m)
    return hot


def spell_events(hot, area_flat, min_dur=MIN_DUR):
    """Vectorised >=min_dur consecutive-hot-day runs per cell -> (durations, areas):
    one entry per cell heat-wave event, with its cell area (km^2)."""
    T = hot.shape[0]
    hct = hot.reshape(T, -1).T.astype("int8")
    h = np.pad(hct, ((0, 0), (1, 1)))
    d = np.diff(h, axis=1)
    cs, ts = np.where(d == 1)
    _, te = np.where(d == -1)
    length = te - ts
    keep = length >= min_dur
    return length[keep], area_flat[cs[keep]]


def active_mask(hot, min_dur=MIN_DUR):
    """Boolean (T, Y, X): True where a cell is inside a >=min_dur consecutive-hot-day
    spell on that day (for the day-of-year timing of heat-wave activity)."""
    T = hot.shape[0]
    hf = hot.reshape(T, -1)
    active = np.zeros_like(hf)
    for c in range(hf.shape[1]):
        col = hf[:, c]
        t = 0
        while t < T:
            if col[t]:
                s = t
                while t < T and col[t]:
                    t += 1
                if t - s >= min_dur:
                    active[s:t, c] = True
            else:
                t += 1
    return active.reshape(hot.shape)


def cell_area_km2(lat, nlon):
    """Spherical cell area per latitude band (km^2) on the model grid."""
    lat = np.asarray(lat, float)
    edges = np.empty(len(lat) + 1)
    edges[1:-1] = (lat[:-1] + lat[1:]) / 2
    edges[0] = lat[0] - (lat[1] - lat[0]) / 2
    edges[-1] = lat[-1] + (lat[-1] - lat[-2]) / 2
    edges = np.clip(edges, -90, 90)
    R = 6371.0088
    dlon = 2 * np.pi / nlon
    return (R ** 2) * dlon * np.abs(np.sin(np.deg2rad(edges[1:]))
                                    - np.sin(np.deg2rad(edges[:-1])))
